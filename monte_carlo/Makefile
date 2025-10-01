# Makefile for Monte Carlo Simulation Engine
# 
# Simple Makefile for the unified Monte Carlo simulation engine

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++11 -I/usr/include/eigen3
LDFLAGS = -lm

# Target executable
TARGET = monte_carlo_sim

# Source files
SOURCES = main.cpp \
          simulation_engine.cpp \
          spin_operations.cpp \
          random.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target - builds the main executable
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build successful! Run with: ./$(TARGET)"

# Compile source files to object files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -f $(OBJECTS) $(TARGET)
	@echo "Clean complete!"

# Declare phony targets
.PHONY: all clean