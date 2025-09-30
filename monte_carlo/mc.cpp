#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>


#include "random.h" 

// Use std:: prefix for standard library components
std::ofstream DATA("DATA.1.dat", std::ios::out);

// Structure for a 2d lattice with coordinates x and y
struct lat_type {
    int x;
    int y;
};

// Global constants and variables
const int size = 16;       // lattice size
const int lsize = size - 1; // array size for lattice
const int n = size * size;  // number of spin points on lattice

float T = 8.0;            // starting point for temperature
const float minT = 0.01;   // minimum temperature
float change = 0.1;       // size of steps for temperature loop

// Nearest neighbor coupling constant
// J > 0: Antiferromagnetic (AFM) coupling - favors antiparallel spins
// J < 0: Ferromagnetic (FM) coupling - favors parallel spins
float J = +1.0;           // coupling strength (default: ferromagnetic)

int lat[size + 1][size + 1]; // 2d lattice for spins

long unsigned int mcs = 20000; // number of Monte Carlo steps
int transient = 10000;          // number of transient steps
double norm = (1.0 / float(mcs * n)); // normalization for averaging
long int seed = -436675; // seed for random number generator

// ---- Forward Declarations of Functions ----
// This tells the compiler that these functions exist and will be defined later.
void initialize(int lat[size + 1][size + 1]);
void output(int lat[size + 1][size + 1]);
void choose_random_pos_lat(lat_type &pos);
int energy_pos(lat_type &pos);
bool test_flip(lat_type pos, int &de);
void flip(lat_type pos);
void transient_results();
int total_magnetization();
int total_energy();
void set_coupling(float new_J);
void print_simulation_info();


// ---- Main Program Entry Point ----
int main() {
    double E = 0, Esq_avg = 0, E_avg = 0, etot = 0, etotsq = 0;
    double M = 0, Msq_avg = 0, M_avg = 0, mtot = 0, mtotsq = 0;
    double Mabs = 0, Mabs_avg = 0, Mq_avg = 0, mabstot = 0, mqtot = 0;
    int de = 0;
    lat_type pos;

    // You can easily change the coupling constant here:
    // set_coupling(-1.0);  // Ferromagnetic (default)
    // set_coupling(1.0);   // Antiferromagnetic
    // set_coupling(-2.0);  // Strong ferromagnetic
    // set_coupling(0.5);   // Weak antiferromagnetic

    initialize(lat);
    print_simulation_info();

    // Write header to data file
    DATA << "# Ising Model Monte Carlo Simulation Results" << std::endl;
    DATA << "# Lattice size: " << size << "x" << size << ", J = " << J << std::endl;
    DATA << "# Columns: T, M_avg, Mabs_avg, Msq_avg, Chi_M, Chi_Mabs, E_avg, Esq_avg, C_v, Binder" << std::endl;
    DATA << "Temperature,Magnetization,AbsMagnetization,MagSqAvg,Susceptibility_M,Susceptibility_Mabs,Energy,EnergySqAvg,SpecificHeat,BinderParameter" << std::endl;

    // Temperature loop
    for (; T >= minT; T = T - change) {
        std::cout << "\nStarting simulation at T = " << T << std::endl;
        transient_results();

        M = total_magnetization();
        Mabs = std::abs(total_magnetization());
        E = total_energy();

        etot = 0; etotsq = 0; mtot = 0; mtotsq = 0; mabstot = 0; mqtot = 0;

        // Monte Carlo loop
        for (int a = 1; a <= mcs; a++) {
            // Print progress every 1000 iterations
            if (a % 10000 == 0) {
                std::cout << "T = " << T << ", Iteration: " << a << "/" << mcs 
                          << " (" << (100.0 * a / mcs) << "%)" << std::endl;
            }
            // Metropolis loop
            for (int b = 1; b <= n; b++) {
                choose_random_pos_lat(pos);
                if (test_flip(pos, de)) {
                    flip(pos);
                    E += 2 * de;
                    M += 2 * lat[pos.x][pos.y];
                }
            }
            Mabs = std::abs(M); // Recalculate absolute magnetization
            etot += E / 2.0;
            etotsq += (E / 2.0) * (E / 2.0);
            mtot += M;
            mtotsq += M * M;
            mqtot += M * M * M * M;
            mabstot += Mabs;
        }

        E_avg = etot * norm;
        Esq_avg = etotsq * norm;
        M_avg = mtot * norm;
        Msq_avg = mtotsq * norm;
        Mabs_avg = mabstot * norm;
        Mq_avg = mqtot * norm;

        DATA << T << "," << M_avg << "," << Mabs_avg << "," << Msq_avg
             << "," << (Msq_avg - (M_avg * M_avg * n)) / (T)
             << "," << (Msq_avg - (Mabs_avg * Mabs_avg * n)) / (T)
             << "," << E_avg << "," << Esq_avg
             << "," << (Esq_avg - (E_avg * E_avg * n)) / (T * T)
             << "," << 1 - ((Mq_avg) / (3 * Msq_avg * Msq_avg)) << std::endl;
        
        std::cout << "Completed T = " << T << " - Results: <M> = " << M_avg 
                  << ", <|M|> = " << Mabs_avg << ", <E> = " << E_avg << std::endl;
    }
    
    std::cout << "\n=== Simulation completed! ===" << std::endl;
    
    return 0;
}


// ---- Definitions for all Functions ----

void initialize(int lat[size + 1][size + 1]) {
    for (int y = size; y >= 1; y--) {
        for (int x = 1; x <= size; x++) {
            if (ran1(&seed) >= 0.5)
                lat[x][y] = 1;
            else
                lat[x][y] = -1;
        }
    }
}

void output(int lat[size + 1][size + 1]) {
    for (int y = size; y >= 1; y--) {
        for (int x = 1; x <= size; x++) {
            if (lat[x][y] < 0)
                std::cout << " - ";
            else
                std::cout << " + ";
        }
        std::cout << std::endl;
    }
}

void choose_random_pos_lat(lat_type &pos) {
    pos.x = (int)ceil(ran1(&seed) * (size));
    pos.y = (int)ceil(ran1(&seed) * (size));
    if (pos.x > size || pos.y > size || pos.x < 1 || pos.y < 1) {
        std::cout << "error in array size allocation for random position on lattice!";
        exit(1);
    }
}

// THIS IS THE FUNCTION THAT WAS LIKELY INCOMPLETE
// Here is the clearly formatted version.
int energy_pos(lat_type &pos) {
    int up, down, left, right;

    // Periodic boundary conditions for y-coordinate
    if (pos.y == size) {
        up = 1;
    } else {
        up = pos.y + 1;
    }
    if (pos.y == 1) {
        down = size;
    } else {
        down = pos.y - 1;
    }

    // Periodic boundary conditions for x-coordinate
    if (pos.x == 1) {
        left = size;
    } else {
        left = pos.x - 1;
    }
    if (pos.x == size) {
        right = 1;
    } else {
        right = pos.x + 1;
    }

    int e = J * lat[pos.x][pos.y] * (lat[left][pos.y] + lat[right][pos.y] + lat[pos.x][up] + lat[pos.x][down]);
    
    return e;
}

bool test_flip(lat_type pos, int &de) {
    de = -2 * energy_pos(pos);
    if (de < 0) {
        return true;
    } else if (ran1(&seed) < std::exp(-de / T)) {
        return true;
    } else {
        return false;
    }
}

void flip(lat_type pos) {
    lat[pos.x][pos.y] = -lat[pos.x][pos.y];
}

void transient_results() {
    std::cout << "Running transient phase (" << transient << " steps)..." << std::endl;
    lat_type pos;
    int de = 0;
    for (int a = 1; a <= transient; a++) {
        // Print progress every 1000 transient steps
        if (a % 1000 == 0) {
            std::cout << "Transient: " << a << "/" << transient 
                      << " (" << (100.0 * a / transient) << "%)" << std::endl;
        }
        for (int b = 1; b <= n; b++) {
            choose_random_pos_lat(pos);
            if (test_flip(pos, de)) {
                flip(pos);
            }
        }
    }
    std::cout << "Transient phase completed." << std::endl;
}

int total_magnetization() {
    int m = 0;
    for (int y = size; y >= 1; y--) {
        for (int x = 1; x <= size; x++) {
            m += lat[x][y];
        }
    }
    return m;
}

int total_energy() {
    lat_type pos;
    int e = 0;
    for (int y = size; y >= 1; y--) {
        pos.y = y;
        for (int x = 1; x <= size; x++) {
            pos.x = x;
            e += energy_pos(pos);
        }
    }
    return e;
}

void set_coupling(float new_J) {
    J = new_J;
    std::cout << "Coupling constant J set to: " << J << std::endl;
    if (J < 0) {
        std::cout << "Ferromagnetic (FM) coupling - favors parallel spins" << std::endl;
    } else if (J > 0) {
        std::cout << "Antiferromagnetic (AFM) coupling - favors antiparallel spins" << std::endl;
    } else {
        std::cout << "Zero coupling - no spin interactions" << std::endl;
    }
}

void print_simulation_info() {
    std::cout << "=== Ising Model Monte Carlo Simulation ===" << std::endl;
    std::cout << "Lattice size: " << size << "x" << size << std::endl;
    std::cout << "Total spins: " << n << std::endl;
    std::cout << "Monte Carlo steps: " << mcs << std::endl;
    std::cout << "Transient steps: " << transient << std::endl;
    std::cout << "Temperature range: " << T << " to " << minT << " (step: " << change << ")" << std::endl;
    std::cout << "Coupling constant J: " << J;
    if (J < 0) {
        std::cout << " (Ferromagnetic)" << std::endl;
    } else if (J > 0) {
        std::cout << " (Antiferromagnetic)" << std::endl;
    } else {
        std::cout << " (No coupling)" << std::endl;
    }
    std::cout << "==========================================" << std::endl;
}