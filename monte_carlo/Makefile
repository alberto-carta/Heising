CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17 -Iinclude -I/usr/include/eigen3
LIBS = -lm

SRCDIR = src
BUILDDIR = build

SOURCES = $(SRCDIR)/simulation_engine.cpp $(SRCDIR)/spin_operations.cpp $(SRCDIR)/random.cpp
IO_SOURCES = $(SRCDIR)/io/configuration_parser.cpp
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o) $(IO_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)

# Create build subdirectories
$(BUILDDIR)/io:
	mkdir -p $(BUILDDIR)/io

all: $(BUILDDIR)/monte_carlo_sim $(BUILDDIR)/unit_tests $(BUILDDIR)/performance_test $(BUILDDIR)/heisenberg_analysis $(BUILDDIR)/ising_analysis $(BUILDDIR)/generic_simulation

$(BUILDDIR)/monte_carlo_sim: $(OBJECTS) $(BUILDDIR)/main.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/unit_tests: $(OBJECTS) $(BUILDDIR)/unit_tests.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/performance_test: $(OBJECTS) $(BUILDDIR)/performance_test.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/heisenberg_analysis: $(OBJECTS) $(BUILDDIR)/heisenberg_analysis.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/ising_analysis: $(OBJECTS) $(BUILDDIR)/ising_analysis.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/generic_simulation: $(OBJECTS) $(BUILDDIR)/generic_simulation.o | $(BUILDDIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR) $(BUILDDIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule for IO subdirectory
$(BUILDDIR)/io/%.o: $(SRCDIR)/io/%.cpp | $(BUILDDIR)/io
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/main.o: examples/main.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/unit_tests.o: tests/unit_tests.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/performance_test.o: benchmarks/performance_test.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/heisenberg_analysis.o: heisenberg_analysis.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/ising_analysis.o: ising_analysis.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/generic_simulation.o: generic_simulation.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

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

clean:
	rm -rf $(BUILDDIR)

.PHONY: all test benchmark heisenberg ising analyze generic ising-config heisenberg-config clean
