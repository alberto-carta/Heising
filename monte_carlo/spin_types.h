/*
 * Spin Types and Models for Monte Carlo Simulations
 * 
 * This file defines different spin types and their operations.
 * We keep it simple and explicit - no fancy templates or inheritance.
 */

#ifndef SPIN_TYPES_H
#define SPIN_TYPES_H

#include <cmath>
#include <iostream>

// Enumeration to choose between different spin models
// Much cleaner than using magic numbers or strings
enum class SpinType {
    ISING,      // Classical Ising model: spins are +1 or -1
    HEISENBERG  // Classical Heisenberg model: 3D unit vectors
};


// This is an enum, and it basically avoids using magic numbers or strings
// SpinType spin_type = SpinType::ISING;  
//
// if (spin_type == SpinType::ISING) {
//     // Do Ising stuff
// } else if (spin_type == SpinType::HEISENBERG) {
//     // Do Heisenberg stuff
// }

// Structure for 2D lattice coordinates
struct lat_type {
    int x;
    int y;
};

// Structure for 3D spin vectors (used in Heisenberg model)
// In Heisenberg model, each spin is a 3D unit vector: S = (Sx, Sy, Sz) with |S| = 1
struct spin3d {
    double x, y, z;
    
    // Constructor to initialize spin
    spin3d(double x = 0.0, double y = 0.0, double z = 1.0) : x(x), y(y), z(z) {}
    
    // Calculate magnitude of spin vector
    double magnitude() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    // Normalize spin to unit length (|S| = 1)
    void normalize() {
        double mag = magnitude();
        if (mag > 1e-10) {  // Avoid division by zero
            x /= mag;
            y /= mag;
            z /= mag;
        }
    }
    
    // Dot product between two spins: S1 · S2
    double dot(const spin3d& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    // Print spin vector (useful for debugging)
    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")";
    }
};

// Function to generate a random unit vector on the sphere
// This is used for proposing new Heisenberg spin orientations
spin3d random_unit_vector();

// Function to generate a small random change to a spin vector
// Used for local updates in Heisenberg model
spin3d small_random_change(const spin3d& current_spin, double max_angle);

#endif // SPIN_TYPES_H