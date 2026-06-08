# Explorers are independent solvers, not front-ends to the C++ DNS

The shared numerical method is implemented three times: the C++ CPU DNS, the
CUDA/MPI HPC backend, and the pure-Python Explorers (`viz/ns2d.py`, `viz/ns3d.py`).
The Explorers deliberately re-implement the method in Python rather than driving
the compiled C++ solver. The reason is interactivity: an Explorer must re-solve
live as the user drags Reynolds/dt sliders and switches scenarios, which needs
the solver state in-process and trivially restartable. A subprocess/IPC bridge
to the C++ binary would add latency and complexity for no teaching benefit. The
trade-off is duplicated numerics: any change to the method must be mirrored in
`cpu_kernels.hpp`, `hpc/cuda_kernels.cu`, `viz/ns2d.py`, and `viz/ns3d.py`. We
accept this because the Python solvers double as the most readable reference
implementation of the math.
