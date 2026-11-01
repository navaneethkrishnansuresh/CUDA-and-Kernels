# Note 02: Threads, Blocks, and Grids

This note explains how CUDA divides one kernel launch into many GPU threads and how each thread identifies the work it owns.

## Index

- blocks and threads in a kernel launch
- CUDA grid, block, and thread hierarchy
- `threadIdx.x` and `blockIdx.x`
- `blockDim.x` and `gridDim.x`
- global thread IDs
- why GPU `printf()` output is not ordered
- mapping threads to positions in a larger problem

Open the notebook: [02_threads_blocks_grids.ipynb](02_threads_blocks_grids.ipynb)
