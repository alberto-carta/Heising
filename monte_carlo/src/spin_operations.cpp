/*
 * Implementation of spin operations for Monte Carlo simulations
 */

#include "spin_types.h"
#include "random.h"
#include <cmath>

extern long int seed;  // Use the same seed as in main program

// Generate a random unit vector uniformly distributed on the unit sphere
// This uses the Marsaglia method - simple and pedagogical
spin3d random_unit_vector() {
    double x1, x2, w;
    
    // Marsaglia method: generate random point in unit circle
    do {
        x1 = 2.0 * ran1(&seed) - 1.0;  // Random number in [-1, 1]
        x2 = 2.0 * ran1(&seed) - 1.0;  // Random number in [-1, 1]
        w = x1*x1 + x2*x2;
    } while (w >= 1.0);  // Reject points outside unit circle
    
    // Convert to 3D unit vector using Marsaglia's formula
    double sqrt_w = std::sqrt(1.0 - w);
    spin3d result;
    result.x = 2.0 * x1 * sqrt_w;
    result.y = 2.0 * x2 * sqrt_w;
    result.z = 1.0 - 2.0 * w;
    
    return result;
}

// Generate a small random change to a spin vector
// This is used for local Monte Carlo updates in Heisenberg model
spin3d small_random_change(const spin3d& current_spin, double max_angle) {
    // Generate a random axis perpendicular to current spin
    spin3d random_axis = random_unit_vector();
    
    // Make sure the random axis is not parallel to current spin
    // If they're too close, generate a new random axis
    double dot_product = std::abs(current_spin.dot(random_axis));
    if (dot_product > 0.95) {  // If vectors are nearly parallel
        // Use a simple perpendicular vector
        if (std::abs(current_spin.z) < 0.9) {
            random_axis = spin3d(0, 0, 1);  // Use z-axis
        } else {
            random_axis = spin3d(1, 0, 0);  // Use x-axis
        }
    }
    
    // Create a vector perpendicular to current spin
    // Use Gram-Schmidt orthogonalization: v_perp = v - (v·s)s
    double projection = random_axis.dot(current_spin);
    spin3d perpendicular;
    perpendicular.x = random_axis.x - projection * current_spin.x;
    perpendicular.y = random_axis.y - projection * current_spin.y;
    perpendicular.z = random_axis.z - projection * current_spin.z;
    perpendicular.normalize();
    
    // Generate random angle in [-max_angle, max_angle]
    double angle = max_angle * (2.0 * ran1(&seed) - 1.0);
    
    // Rotate current spin by this angle around the perpendicular axis
    // New spin = cos(angle) * current + sin(angle) * perpendicular
    spin3d new_spin;
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);
    
    new_spin.x = cos_angle * current_spin.x + sin_angle * perpendicular.x;
    new_spin.y = cos_angle * current_spin.y + sin_angle * perpendicular.y;
    new_spin.z = cos_angle * current_spin.z + sin_angle * perpendicular.z;
    
    // Ensure it's normalized (should be already, but numerical precision...)
    new_spin.normalize();
    
    return new_spin;
}