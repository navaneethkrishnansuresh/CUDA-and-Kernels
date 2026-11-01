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

## What CUDA Covers Here

- host and device responsibilities
- CUDA C++ source, compilation, and kernel launches
- threads, blocks, grids, and global indexing
- GPU memory allocation and CPU to GPU data movement
- synchronization and kernel correctness
- building kernels and understanding their performance behavior

## Later kernel work

Later notes will move from basic CUDA programs into CUDA implementations of model operations: reductions, matrix operations, RMSNorm, softmax, sampling, and attention building blocks. They will be added once the lower-level CUDA pieces needed to understand them are in place.

## Disclaimer

This is my personal implementation work around CUDA programming. Take inspiration from it, use it to understand a few things, and adapt ideas for your own experiments.

It is not an official CUDA reference, a production kernel library, or a claim of a new optimization method. Some examples are intentionally simple so the underlying behavior is visible before performance work is introduced.

Please skip this for now if you do not already have a solid grasp of C++ and the basic idea of how GPUs execute work. If you are a larper who only wants to skim keywords, this repo is not for you. It is meant for people who want to read the code, run it, and understand what each note does.

Notes may be added, split, or reordered as the implementation grows. New notes will be added every one to two days when possible.

## Notes

| Note | Topic | Status |
| --- | --- | --- |
| 01 | [First CUDA Program](01_first_cuda_program/) | Available |
| 02 | [Threads, Blocks, and Grids](02_threads_blocks_grids/) | Available |
| 03 | Vector Addition | Not available yet |

More notes coming soon.

## Running the notebooks

Use a CUDA-enabled environment with `nvcc` available. The notebooks are written to run in Google Colab with a GPU runtime. Select a GPU runtime before running them.
