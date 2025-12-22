# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17 -Iinclude -I/usr/include/eigen3
LIBS = -lm

# MPI detection and settings
MPI_CXX := $(shell command -v mpic++ 2> /dev/null)
ifdef MPI_CXX
    HAS_MPI = 1
    # Force mpic++ to use g++ as the underlying compiler
    MPI_COMPILE = OMPI_CXX=g++ mpic++
    MPI_CXXFLAGS = $(CXXFLAGS) -DUSE_MPI
    MPI_LIBS = $(LIBS)
    $(info MPI detected: mpic++ available)
else
    HAS_MPI = 0
    $(info MPI not detected: building serial version only)
endif

SRCDIR = src
BUILDDIR = build

SOURCES = $(SRCDIR)/simulation_engine.cpp $(SRCDIR)/spin_operations.cpp $(SRCDIR)/random.cpp $(SRCDIR)/simulation_utils.cpp $(SRCDIR)/profiling.cpp
IO_SOURCES = $(SRCDIR)/io/configuration_parser.cpp $(SRCDIR)/io/diagnostic_utils.cpp
MPI_SOURCES = $(SRCDIR)/mpi_wrapper.cpp
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o) $(IO_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)
MPI_OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/mpi/%.o) $(IO_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/mpi/%.o) $(MPI_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/mpi/%.o)

# Create build subdirectories
$(BUILDDIR)/io:
	mkdir -p $(BUILDDIR)/io

$(BUILDDIR)/mpi:
	mkdir -p $(BUILDDIR)/mpi/io

# Default target
ifeq ($(HAS_MPI),1)
all: tests heising
else
all: tests
	@echo "WARNING: MPI not available - heising.x will not be built"
endif

# Main heising executable (requires MPI)
ifeq ($(HAS_MPI),1)
heising: $(BUILDDIR)/heising.x
else
heising:
	@echo "ERROR: MPI required to build heising.x"
	@exit 1
endif

# Test executables (serial, no MPI required)
tests: $(BUILDDIR)/unit_tests $(BUILDDIR)/multi_atom_cell_tests

# Legacy executables (kept for compatibility)
legacy: $(BUILDDIR)/monte_carlo_sim $(BUILDDIR)/performance_test $(BUILDDIR)/heisenberg_analysis $(BUILDDIR)/ising_analysis

# Main heising executable (MPI-enabled)
$(BUILDDIR)/heising.x: $(MPI_OBJECTS) $(BUILDDIR)/mpi/heising_main.o | $(BUILDDIR)/mpi
	$(MPI_COMPILE) $^ -o $@ $(MPI_LIBS)

# Test executables (serial)
$(BUILDDIR)/monte_carlo_sim: $(OBJECTS) $(BUILDDIR)/main.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/unit_tests: $(OBJECTS) $(BUILDDIR)/unit_tests.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/multi_atom_cell_tests: $(OBJECTS) $(BUILDDIR)/multi_atom_cell_tests.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/performance_test: $(OBJECTS) $(BUILDDIR)/performance_test.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/heisenberg_analysis: $(OBJECTS) $(BUILDDIR)/heisenberg_analysis.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/ising_analysis: $(OBJECTS) $(BUILDDIR)/ising_analysis.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/generic_simulation: $(OBJECTS) $(BUILDDIR)/generic_simulation.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

# MPI executable
$(BUILDDIR)/generic_simulation_mpi: $(MPI_OBJECTS) $(BUILDDIR)/mpi/generic_simulation.o | $(BUILDDIR)/mpi
	$(MPI_COMPILE) $^ -o $@ $(MPI_LIBS)

# Serial object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR) $(BUILDDIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule for IO subdirectory
$(BUILDDIR)/io/%.o: $(SRCDIR)/io/%.cpp | $(BUILDDIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

# MPI object files
$(BUILDDIR)/mpi/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)/mpi
	$(MPI_COMPILE) $(MPI_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/mpi/io/%.o: $(SRCDIR)/io/%.cpp | $(BUILDDIR)/mpi
	$(MPI_COMPILE) $(MPI_CXXFLAGS) -c $< -o $@

$(BUILDDIR)/main.o: examples/main.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/unit_tests.o: tests/unit_tests.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/multi_atom_cell_tests.o: tests/multi_atom_cell_tests.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Legacy executables
$(BUILDDIR)/performance_test.o: benchmarks/performance_test.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/heisenberg_analysis.o: heisenberg_analysis.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/ising_analysis.o: ising_analysis.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# MPI heising_main object
$(BUILDDIR)/mpi/heising_main.o: heising_main.cpp | $(BUILDDIR)/mpi
	$(MPI_COMPILE) $(MPI_CXXFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

test: $(BUILDDIR)/unit_tests
	./$(BUILDDIR)/unit_tests

benchmark: $(BUILDDIR)/performance_test
	./$(BUILDDIR)/performance_test

heisenberg: $(BUILDDIR)/heisenberg_analysis
	./$(BUILDDIR)/heisenberg_analysis

ising: $(BUILDDIR)/ising_analysis
	./$(BUILDDIR)/ising_analysis

analyze: $(BUILDDIR)/heisenberg_analysis $(BUILDDIR)/ising_analysis
	@echo "Running Heisenberg ferromagnet analysis..."
	./$(BUILDDIR)/heisenberg_analysis
	@echo ""
	@echo "Running Ising ferromagnet analysis..."
	./$(BUILDDIR)/ising_analysis

# Run generic simulation with configuration files
generic: $(BUILDDIR)/generic_simulation
	./$(BUILDDIR)/generic_simulation ../examples/simulation.toml

ising-config: $(BUILDDIR)/generic_simulation
	./$(BUILDDIR)/generic_simulation ../examples/ising_ferromagnet.toml

heisenberg-config: $(BUILDDIR)/generic_simulation  
	./$(BUILDDIR)/generic_simulation ../examples/heisenberg_ferromagnet.toml

# MPI targets (only work if MPI is available)
ifeq ($(HAS_MPI),1)
mpi-help:
	@echo "MPI is available! Use these commands:"
	@echo "  mpirun -np <N> ./build/generic_simulation_mpi <config.toml>"
	@echo ""
	@echo "Example:"
	@echo "  mpirun -np 4 ./build/generic_simulation_mpi examples/heisenberg_ferromagnet/simulation.toml"
else
mpi-help:
	@echo "MPI is not available. Install OpenMPI to use parallel features:"
	@echo "  sudo apt install libopenmpi-dev openmpi-bin"
endif

clean:
	rm -rf $(BUILDDIR)

.PHONY: all serial mpi test benchmark heisenberg ising analyze generic ising-config heisenberg-config mpi-help clean
