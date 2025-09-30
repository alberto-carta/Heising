/*
 * Simple Random Number Generation using C++ Standard Library
 * 
 * This replaces the complex Park-Miller generator with modern C++ random facilities.
 * Much simpler and easier to understand for learning purposes!
 */

#include <random>

// Global random number generator state
// Using static to keep state between function calls, just like the old ran1
static std::mt19937 rng;  // Mersenne Twister generator (high quality, standard)
static std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);  // [0,1) distribution
static bool initialized = false;

/*
 * Simple replacement for ran1 using C++ standard library
 * 
 * Input:  idum - pointer to seed (for compatibility with original interface)
 *         - Use negative value to initialize/reinitialize
 * Output: Random float in range [0, 1)
 * 
 * Much simpler than the original Park-Miller generator!
 */
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
/*
 * That's it! Much simpler than the original 80+ lines of complex code.
 * 
 * Benefits of using standard library:
 * 1. Well-tested and high-quality random number generation
 * 2. Much easier to understand and maintain
 * 3. Portable across different systems
 * 4. Same interface as original ran1 function
 * 
 * The Mersenne Twister (std::mt19937) is an excellent generator with:
 * - Very long period (2^19937 - 1)
 * - Good statistical properties
 * - Fast generation
 * - Standard in modern C++
 */