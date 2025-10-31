/*
 * Random Number Generation Header
 * 
 * Simple replacement for the original ran1 function using C++ standard library.
 * This header contains only the declaration - implementation is in random.cpp
 */

#ifndef RANDOM_H
#define RANDOM_H

/*
 * Simple replacement for ran1 using C++ standard library
 * 
 * Input:  idum - pointer to seed (for compatibility with original interface)
 *         - Use negative value to initialize/reinitialize
 * Output: Random float in range [0, 1)
 */
float ran1(long *idum);

#endif // RANDOM_H