cmake_minimum_required(VERSION 3.18)
project(ns3d_cpu LANGUAGES CXX)

# ------------------------------------------------------------------
#  CPU-only 3D pseudo-spectral Navier-Stokes DNS.
#  Runs on an ordinary multi-core machine (FFTW3 + OpenMP).
#  The GPU / multi-GPU MPI backend lives in ./hpc and builds separately.
# ------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

set(NS3D_N 64 CACHE STRING "Grid resolution per axis (compile-time constant)")

find_package(OpenMP REQUIRED)
find_library(FFTW3_LIB     fftw3     REQUIRED)
find_library(FFTW3_OMP_LIB fftw3_omp REQUIRED)

add_executable(ns3d_cpu solver_cpu.cpp)
target_compile_definitions(ns3d_cpu PRIVATE NS3D_N=${NS3D_N})
target_link_libraries(ns3d_cpu PRIVATE
    OpenMP::OpenMP_CXX
    ${FFTW3_OMP_LIB}
    ${FFTW3_LIB}
    m)

message(STATUS "ns3d_cpu: N=${NS3D_N}, build=${CMAKE_BUILD_TYPE}")
