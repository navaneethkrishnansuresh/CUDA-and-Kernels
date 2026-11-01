# Note 03: Vector Addition

This note turns CUDA's execution model into a complete program: data starts in CPU memory, moves to GPU memory, is processed by one kernel, and returns to the CPU.

## Index

- independent vector addition work
- host arrays and device pointers
- GPU memory allocation with `cudaMalloc()`
- CPU to GPU and GPU to CPU copies with `cudaMemcpy()`
- the difference between `d_a` and `&d_a`
- global thread indexing
- bounds checks for extra threads
- ceiling division for block counts
- kernel launch, synchronization, and freeing GPU memory

Open the notebook: [03_vector_addition.ipynb](03_vector_addition.ipynb)
