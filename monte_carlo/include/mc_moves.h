/*
 * Monte Carlo Move Proposer Header
 * 
 * Separates move proposal logic from simulation engine
 */

#pragma once

#include "simulation_engine.h"
#include "multi_spin.h"
#include "spin_types.h"
#include <vector>

// Forward declaration
class MonteCarloSimulation;

// MoveProposal struct is defined in simulation_engine.h

// Move proposer class - handles move proposals
class MoveProposer {
private:
    MonteCarloSimulation& sim;
    
public:
    MoveProposer(MonteCarloSimulation& simulation);
    
    // Propose single-spin moves
    MoveProposal propose_ising_flip(int x, int y, int z, int spin_id);
    MoveProposal propose_heisenberg_rotation(int x, int y, int z, int spin_id);
    MoveProposal propose_heisenberg_flip(int x, int y, int z, int spin_id);
    
    // Propose site-level move (flips all spins at a site)
    MoveProposal propose_site_double_tunnel(int x, int y, int z, int site_id);
    
    // Propose slab tunnel move (transforms entire slab from pattern1 to pattern2)
    MoveProposal propose_slab_tunnel(int x_start, int y_start, int z_start);
};

