# CUDA and Kernels

This repository documents CUDA programming from first principles.

These notes cover CUDA programming from the basic building blocks of CUDA C++ through GPU execution, memory, kernels, and performance work.

The goal is to understand what actually happens when code runs on a GPU: how the host launches device work, how kernels map work to threads, how data moves between CPU and GPU memory, and why GPU programs need to be structured differently from normal CPU programs.

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

Everything is implementation-first. Each note pairs a direct explanation with a small CUDA C++ program that can be compiled and run.

## Disclaimer

This is my personal implementation work around CUDA programming. Take inspiration from it, use it to understand a few things, and adapt ideas for your own experiments.

It is not an official CUDA reference, a production kernel library, or a claim of a new optimization method. Some examples are intentionally simple so the underlying behavior is visible before performance work is introduced.

Please skip this for now if you do not have basic C++, programming, and computer-systems knowledge. If you are a larper who only wants to skim keywords, this repo is not for you. It is meant for people who want to read the code, run it, and understand what each note does.

Notes may be added, split, or reordered as the implementation grows. New notes will be added every one to two days when possible.

## Notes

| Note | Topic | Status |
| --- | --- | --- |
| 01 | [First CUDA Program](01_first_cuda_program/01_first_cuda_program.ipynb) | Available |
| 02 | [Threads, Blocks, and Grids](02_threads_blocks_grids/02_threads_blocks_grids.ipynb) | Available |
| 03 | Vector Addition | Not available yet |

More notes coming soon.

## Running the notebooks

Use a CUDA-enabled environment with `nvcc` available. The notebooks are written to run in Google Colab with a GPU runtime. Select a GPU runtime before running them.
