# Heising - Monte Carlo Simulation for Magnetic Systems
# Minimal CMake configuration

cmake_minimum_required(VERSION 3.10)
project(Heising VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# ============================================================================
# REQUIRED DEPENDENCIES
# ============================================================================
# This project requires:
#   - MPI (Message Passing Interface)
#   - Eigen3 (Linear algebra library)
#   - toml11 (TOML parser, header-only)
#
# Installation instructions:
#   Ubuntu/Debian:
#     sudo apt install libopenmpi-dev libeigen3-dev
#     git clone https://github.com/ToruNiina/toml11.git (header-only)
#
#   Fedora/RHEL:
#     sudo dnf install openmpi-devel eigen3-devel
#
#   macOS:
#     brew install open-mpi eigen
# ============================================================================

# Find MPI (required)
find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR 
        "MPI not found!\n"
        "Please install MPI:\n"
        "  Ubuntu/Debian: sudo apt install libopenmpi-dev\n"
        "  Fedora/RHEL:   sudo dnf install openmpi-devel\n"
        "  macOS:         brew install open-mpi"
    )
endif()

# Find Eigen3 (required)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Try alternative location
    if(EXISTS "/usr/include/eigen3")
        set(EIGEN3_FOUND TRUE)
        set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
        message(STATUS "Eigen3 found at /usr/include/eigen3")
        include_directories(/usr/include/eigen3)
    else()
        message(FATAL_ERROR 
            "Eigen3 not found!\n"
            "Please install Eigen3:\n"
            "  Ubuntu/Debian: sudo apt install libeigen3-dev\n"
            "  Fedora/RHEL:   sudo dnf install eigen3-devel\n"
            "  macOS:         brew install eigen"
        )
    endif()
else()
    set(EIGEN3_FOUND TRUE)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Check for toml11 (header-only library)
find_path(TOML11_INCLUDE_DIR toml.hpp
    PATHS
        /usr/include
        /usr/local/include
        ${CMAKE_SOURCE_DIR}/../toml11/include
        ${CMAKE_SOURCE_DIR}/external/toml11/include
)

if(NOT TOML11_INCLUDE_DIR)
    message(FATAL_ERROR 
        "toml11 not found!\n"
        "toml11 is a header-only library. Please install it:\n"
        "  Option 1: git clone https://github.com/ToruNiina/toml11.git\n"
        "            Then add -DTOML11_INCLUDE_DIR=<path>/toml11/include to cmake\n"
        "  Option 2: Copy toml.hpp to /usr/local/include/\n"
        "  Option 3: Place toml11/ in Heising/toml11/"
    )
else()
    message(STATUS "toml11 found: ${TOML11_INCLUDE_DIR}")
    include_directories(${TOML11_INCLUDE_DIR})
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

# ============================================================================
# SOURCE FILES
# ============================================================================

# Collect all source files
file(GLOB CORE_SOURCES 
    "src/simulation_engine.cpp"
    "src/spin_operations.cpp"
    "src/random.cpp"
    "src/simulation_utils.cpp"
    "src/profiling.cpp"
)

file(GLOB IO_SOURCES 
    "src/io/configuration_parser.cpp"
    "src/io/diagnostic_utils.cpp"
)

set(MPI_SOURCE src/mpi_wrapper.cpp)

set(ALL_SOURCES ${CORE_SOURCES} ${IO_SOURCES})

# ============================================================================
# MAIN EXECUTABLE (MPI-enabled)
# ============================================================================

add_executable(heising.x 
    heising_main.cpp 
    ${ALL_SOURCES}
    ${MPI_SOURCE}
)

target_compile_definitions(heising.x PRIVATE USE_MPI)
target_link_libraries(heising.x 
    ${MPI_CXX_LIBRARIES}
    m
)

# Add custom target 'heising' as alias
add_custom_target(heising DEPENDS heising.x)

# ============================================================================
# TESTS
# ============================================================================

# Unit tests
add_executable(unit_tests 
    tests/unit_tests.cpp 
    ${ALL_SOURCES}
    ${MPI_SOURCE}
)
target_link_libraries(unit_tests m)

# IO tests
add_executable(io_tests 
    tests/io_tests.cpp 
    ${ALL_SOURCES}
    ${MPI_SOURCE}
)
target_compile_definitions(io_tests PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/io_test_data")
target_link_libraries(io_tests m)

# Custom target to build all tests
add_custom_target(tests DEPENDS unit_tests io_tests)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running unit tests..."
    COMMAND ${CMAKE_BINARY_DIR}/unit_tests
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Running IO tests..."
    COMMAND ${CMAKE_BINARY_DIR}/io_tests
    DEPENDS tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running all tests..."
)

# ============================================================================
# INSTALLATION (optional)
# ============================================================================

install(TARGETS heising.x DESTINATION bin)

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Heising Configuration Summary")
message(STATUS "========================================")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags:        ${CMAKE_CXX_FLAGS}")
message(STATUS "MPI found:        ${MPI_FOUND}")
message(STATUS "MPI compiler:     ${MPI_CXX_COMPILER}")
message(STATUS "Eigen3 found:     ${EIGEN3_FOUND}")
message(STATUS "Eigen3 location:  ${EIGEN3_INCLUDE_DIR}")
message(STATUS "toml11 location:  ${TOML11_INCLUDE_DIR}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  make heising     - Build main MPI executable")
message(STATUS "  make tests       - Build all tests")
message(STATUS "  make run_tests   - Build and run all tests")
message(STATUS "========================================")
message(STATUS "")
