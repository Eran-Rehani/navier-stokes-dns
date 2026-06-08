# ================================================================
#  Makefile — 3D Pseudo-Spectral Navier-Stokes DNS
#
#  Build:   make -j4
#  Clean:   make clean
#
#  Environment requirements:
#    - NVCC (CUDA Toolkit ≥ 11.0)
#    - MPI with CUDA-aware support (OpenMPI ≥ 4.1, MVAPICH2-GDR)
#    - cuFFT (included in CUDA Toolkit)
#
#  Override defaults on the command line:
#    make CUDA_ARCH="-arch=sm_90" NS3D_N=512
# ================================================================

# ---- Compilers ----
NVCC     := nvcc
MPICXX   := mpicxx

# ---- Architecture ----
CUDA_ARCH ?= -arch=sm_80

# ---- Grid resolution (compile-time constant) ----
NS3D_N ?= 256

# ---- Flags ----
CXX_STD    := -std=c++20
OPT        := -O3 --use_fast_math
DEFINES    := -DNS3D_N=$(NS3D_N)

# Extract MPI compiler/linker flags
MPI_CFLAGS  := $(shell $(MPICXX) --showme:compile 2>/dev/null || echo "")
MPI_LDFLAGS := $(shell $(MPICXX) --showme:link    2>/dev/null || echo "")

# nvcc passes host flags via -Xcompiler
NVCC_FLAGS := $(CXX_STD) $(OPT) $(CUDA_ARCH) $(DEFINES) \
              -Xcompiler "$(MPI_CFLAGS)"

LDFLAGS := $(MPI_LDFLAGS) -lcufft -lcudart

# ---- Sources and objects ----
TARGET := ns3d_solver

SRCS_CU  := cuda_kernels.cu
SRCS_CPP := decomposition.cpp solver.cpp

OBJS := cuda_kernels.o decomposition.o solver.o

# ---- Rules ----

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

cuda_kernels.o: cuda_kernels.cu cuda_kernels.cuh common.hpp
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

decomposition.o: decomposition.cpp decomposition.hpp common.hpp cuda_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

solver.o: solver.cpp common.hpp decomposition.hpp cuda_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
