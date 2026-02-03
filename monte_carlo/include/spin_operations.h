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

// Select a random small rotation angle in [-max_angle, max_angle]
double select_small_angle(double max_angle);

// Apply a rotation to a spin by a given angle around a perpendicular axis
// The rotation axis is chosen randomly but perpendicular to the current spin
spin3d apply_rotation(const spin3d& original_spin, double angle);

// Flip a Heisenberg spin (reverse its direction: S -> -S)
// This is a non-local move that can help escape local minima
spin3d flip_heisenberg_spin(const spin3d& original_spin);

#endif // SPIN_OPERATIONS_H