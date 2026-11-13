# Note 05: 2D and 3D CUDA Indexing

This note extends CUDA's one-dimensional global indexing model to two and three dimensions. It maps threads to matrix rows and columns, image-like grids, and 3D volumes while keeping the underlying memory layout explicit.

## Index

- `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` as 3D coordinates
- 2D thread blocks and grids with `dim3`
- global row and column calculation
- matrix storage in linear row-major memory
- mapping `(row, col)` to a linear index
- a 2D matrix-addition kernel with boundary checks
- 3D coordinates, grids, and linear memory addressing

Open the notebook: [05_2d_3d_cuda_indexing.ipynb](05_2d_3d_cuda_indexing.ipynb)
