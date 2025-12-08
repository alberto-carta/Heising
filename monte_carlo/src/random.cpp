#include "random.h"
#include <random>

// Global random number generator state
static std::mt19937 rng;  // Mersenne Twister generator 
static std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);  // [0,1) distribution
static bool initialized = false;

float ran1(long *idum)
{
    // Initialize generator if not done yet or if seed is negative (reset)
    if (!initialized || *idum < 0) {
        // Use absolute value of seed for initialization
        unsigned int seed = (*idum < 0) ? -(*idum) : *idum;
        if (seed == 0) seed = 1;  // Avoid seed = 0
        
        rng.seed(seed);  // Initialize Mersenne Twister with seed
        initialized = true;
        
        // Make seed positive for compatibility
        if (*idum <= 0) *idum = seed;
    }
    
    // Generate and return random number in [0, 1)
    return uniform_dist(rng);
}
