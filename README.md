# CUDA and Kernels

<p align="center">
  <img src="https://cdn.simpleicons.org/nvidia/76B900" alt="NVIDIA CUDA" width="72" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-C%2B%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA C++" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab" />
</p>

This repository documents CUDA programming from first principles.

These notes cover CUDA programming from the basic building blocks of CUDA C++ through GPU execution, memory, kernels, and performance work.

They explain how the host launches device work, how kernels map work to threads, how data moves between CPU and GPU memory, and why GPU programs need to be structured differently from normal CPU programs.

The notes start from small runnable pieces and build upward:

```text
CUDA C++ source
→ host and device
→ kernel launch
→ threads, blocks, and grids
→ global indexing
→ GPU memory
→ CUDA kernels
→ performance work
```

Every released note has an explanation and a small CUDA C++ program that can be compiled and run.

## Topic map

These are the topics this repository will cover. They are not one-notebook-per-topic sections. One topic can contain several notes, and CUDA Basics will grow beyond the first three notes as needed.

1. CUDA Basics
2. GPU Architecture and Execution
3. CUDA Memory Spaces and Data Movement
4. Global Memory, Coalescing, and Caches
5. Shared Memory and Tiling
6. Warps, Divergence, Registers, and Occupancy
7. Synchronization, Atomics, and Memory Ordering
8. Parallel Algorithm Patterns
9. CUDA Debugging, Correctness, and Testing
10. GEMM: Naive to Optimized
11. Tensor Cores, Precision, and Quantization
12. Streams, Async Pipelines, and CUDA Graphs
13. Nsight and Performance Engineering
14. CUDA Compiler, PTX, and SASS
15. Modern NVIDIA Architecture Features
16. Transformer Kernels
17. Attention and LLM GPU Performance
18. CUDA Inside Inference Runtimes: Bridge to the Inference Project
19. Triton, CUTLASS/CuTe, and PyTorch Extensions
20. Multi-GPU, Interconnects, and Collectives

## Later kernel work

Later notes will move from basic CUDA programs into CUDA implementations of model operations: reductions, matrix operations, RMSNorm, softmax, sampling, and attention building blocks. They will be added once the lower-level CUDA pieces needed to understand them are in place.

## Disclaimer

This is my personal implementation work around CUDA programming. Take inspiration from it, use it to understand a few things, and adapt ideas for your own experiments.

It is not an official CUDA reference, a production kernel library, or a claim of a new optimization method. Some examples are intentionally simple so the underlying behavior is visible before performance work is introduced.

Please skip this for now if you do not already have a solid grasp of C++ and the basic idea of how GPUs execute work. If you are a larper who only wants to skim keywords, this repo is not for you. It is meant for people who want to read the code, run it, and understand what each note does.

Notes may be added, split, or reordered as the implementation grows. New notes will be added every one to two days when possible.

## CUDA Basics notes

The notes below belong to Topic 1, CUDA Basics. More notes may be added here before the repository moves to GPU Architecture and Execution.

| Note | Topic | Status |
| --- | --- | --- |
| 01 | [First CUDA Program](01_first_cuda_program/) | Available |
| 02 | [Threads, Blocks, and Grids](02_threads_blocks_grids/) | Available |
| 03 | Vector Addition | Not available yet |

More notes coming soon.

## Running the notebooks

Use a CUDA-enabled environment with `nvcc` available. The notebooks are written to run in Google Colab with a GPU runtime. Select a GPU runtime before running them.
