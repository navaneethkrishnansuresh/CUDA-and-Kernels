# Note 04: Scaling CUDA Kernels with Grid-Stride Loops

This note takes vector addition beyond one thread per element. It shows how one kernel can handle an arbitrary input size, including when the launched grid has fewer threads than elements.

## Index

- ceiling division for the number of blocks
- bounds checks for arbitrary input sizes
- multi-block execution
- grid-stride loops
- choosing a grid size independently of the input size

Open the notebook: [04_scaling_cuda_kernels.ipynb](04_scaling_cuda_kernels.ipynb)
