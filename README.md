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

The topic map and the note lists within it may change as the repository develops. Topics and subtopics can be added, updated, split, or merged when that makes the implementation easier to follow.

| Topic | Status |
| --- | --- |
| [1. CUDA Basics](topics/01_cuda_basics/) | Available |
| [2. GPU Architecture and Execution](topics/02_gpu_architecture_execution/) | Not available yet |
| [3. CUDA Memory Spaces and Data Movement](topics/03_cuda_memory_spaces_data_movement/) | Not available yet |
| [4. Global Memory, Coalescing, and Caches](topics/04_global_memory_coalescing_caches/) | Not available yet |
| [5. Shared Memory and Tiling](topics/05_shared_memory_tiling/) | Not available yet |
| [6. Warps, Divergence, Registers, and Occupancy](topics/06_warps_divergence_registers_occupancy/) | Not available yet |
| [7. Synchronization, Atomics, and Memory Ordering](topics/07_synchronization_atomics_memory_ordering/) | Not available yet |
| [8. Parallel Algorithm Patterns](topics/08_parallel_algorithm_patterns/) | Not available yet |
| [9. CUDA Debugging, Correctness, and Testing](topics/09_cuda_debugging_correctness_testing/) | Not available yet |
| [10. GEMM: Naive to Optimized](topics/10_gemm_naive_to_optimized/) | Not available yet |
| [11. Tensor Cores, Precision, and Quantization](topics/11_tensor_cores_precision_quantization/) | Not available yet |
| [12. Streams, Async Pipelines, and CUDA Graphs](topics/12_streams_async_pipelines_cuda_graphs/) | Not available yet |
| [13. Nsight and Performance Engineering](topics/13_nsight_performance_engineering/) | Not available yet |
| [14. CUDA Compiler, PTX, and SASS](topics/14_cuda_compiler_ptx_sass/) | Not available yet |
| [15. Modern NVIDIA Architecture Features](topics/15_modern_nvidia_architecture_features/) | Not available yet |
| [16. Transformer Kernels](topics/16_transformer_kernels/) | Not available yet |
| [17. Attention and LLM GPU Performance](topics/17_attention_llm_gpu_performance/) | Not available yet |
| [18. CUDA Inside Inference Runtimes](topics/18_cuda_inside_inference_runtimes/) | Not available yet |
| [19. Triton, CUTLASS/CuTe, and PyTorch Extensions](topics/19_triton_cutlass_cute_pytorch_extensions/) | Not available yet |
| [20. Multi-GPU, Interconnects, and Collectives](topics/20_multi_gpu_interconnects_collectives/) | Not available yet |

## Later kernel work

Later notes will move from basic CUDA programs into CUDA implementations of model operations: reductions, matrix operations, RMSNorm, softmax, sampling, and attention building blocks. They will be added once the lower-level CUDA pieces needed to understand them are in place.

## Disclaimer

This is my personal implementation work around CUDA programming. Take inspiration from it, use it to understand a few things, and adapt ideas for your own experiments.

It is not an official CUDA reference, a production kernel library, or a claim of a new optimization method. Some examples are intentionally simple so the underlying behavior is visible before performance work is introduced.

Please skip this for now if you do not already have a solid grasp of C++ and the basic idea of how GPUs execute work. If you are a larper who only wants to skim keywords, this repo is not for you. It is meant for people who want to read the code, run it, and understand what each note does.

Notes may be added, split, or reordered as the implementation grows. New notes will be added every one to two days when possible.

## Running the notebooks

Use a CUDA-enabled environment with `nvcc` available. The notebooks are written to run in Google Colab with a GPU runtime. Select a GPU runtime before running them.
