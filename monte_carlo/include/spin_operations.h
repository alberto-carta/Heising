/*
 * Spin Operations Header
 * 
 * This file declares functions for 3D spin vector operations,
 * particularly for the Heisenberg model.
 */

#ifndef SPIN_OPERATIONS_H
#define SPIN_OPERATIONS_H

#include "spin_types.h"

// Generate a random unit vector uniformly distributed on the unit sphere
// Uses the Marsaglia method for uniform sampling
spin3d random_unit_vector();

// Apply a small random change to a spin vector (for local Monte Carlo moves)
// The spin is rotated by a small random angle around a random axis
// max_angle: maximum rotation angle in radians
spin3d small_random_change(const spin3d& original_spin, double max_angle);

#endif // SPIN_OPERATIONS_H