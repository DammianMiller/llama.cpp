# Param sweep B — batch/ubatch at d=0

Model: Qwen3.6-35B-A3B-UD-IQ4_XS | ngl 99, fa 1, ctk/ctv q8_0, t 16

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 49152 | int dot: 0 | matrix cores: NV_coopmat2
llama-bench: benchmark 1/12: starting
llama-bench: benchmark 1/12: warmup prompt run
llama-bench: benchmark 1/12: prompt run 1/5
llama-bench: benchmark 1/12: prompt run 2/5
llama-bench: benchmark 1/12: prompt run 3/5
llama-bench: benchmark 1/12: prompt run 4/5
llama-bench: benchmark 1/12: prompt run 5/5
| model                          |       size |     params | backend    | ngl | n_batch | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |      512 |   q8_0 |   q8_0 |  1 |           pp512 |      2502.55 ± 58.19 |
llama-bench: benchmark 2/12: starting
llama-bench: benchmark 2/12: warmup generation run
llama-bench: benchmark 2/12: generation run 1/5
llama-bench: benchmark 2/12: generation run 2/5
llama-bench: benchmark 2/12: generation run 3/5
llama-bench: benchmark 2/12: generation run 4/5
llama-bench: benchmark 2/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |      512 |   q8_0 |   q8_0 |  1 |           tg128 |        112.30 ± 0.93 |
llama-bench: benchmark 3/12: starting
llama-bench: benchmark 3/12: warmup prompt run
llama-bench: benchmark 3/12: prompt run 1/5
llama-bench: benchmark 3/12: prompt run 2/5
llama-bench: benchmark 3/12: prompt run 3/5
llama-bench: benchmark 3/12: prompt run 4/5
llama-bench: benchmark 3/12: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |     1024 |   q8_0 |   q8_0 |  1 |           pp512 |      2581.06 ± 31.12 |
llama-bench: benchmark 4/12: starting
llama-bench: benchmark 4/12: warmup generation run
llama-bench: benchmark 4/12: generation run 1/5
llama-bench: benchmark 4/12: generation run 2/5
llama-bench: benchmark 4/12: generation run 3/5
llama-bench: benchmark 4/12: generation run 4/5
llama-bench: benchmark 4/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |     1024 |   q8_0 |   q8_0 |  1 |           tg128 |        112.29 ± 1.48 |
llama-bench: benchmark 5/12: starting
llama-bench: benchmark 5/12: warmup prompt run
llama-bench: benchmark 5/12: prompt run 1/5
llama-bench: benchmark 5/12: prompt run 2/5
llama-bench: benchmark 5/12: prompt run 3/5
llama-bench: benchmark 5/12: prompt run 4/5
llama-bench: benchmark 5/12: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |     2048 |   q8_0 |   q8_0 |  1 |           pp512 |      2503.56 ± 60.95 |
llama-bench: benchmark 6/12: starting
llama-bench: benchmark 6/12: warmup generation run
llama-bench: benchmark 6/12: generation run 1/5
llama-bench: benchmark 6/12: generation run 2/5
llama-bench: benchmark 6/12: generation run 3/5
llama-bench: benchmark 6/12: generation run 4/5
llama-bench: benchmark 6/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |     2048 |   q8_0 |   q8_0 |  1 |           tg128 |        112.37 ± 1.69 |
llama-bench: benchmark 7/12: starting
llama-bench: benchmark 7/12: warmup prompt run
llama-bench: benchmark 7/12: prompt run 1/5
llama-bench: benchmark 7/12: prompt run 2/5
llama-bench: benchmark 7/12: prompt run 3/5
llama-bench: benchmark 7/12: prompt run 4/5
llama-bench: benchmark 7/12: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |      512 |   q8_0 |   q8_0 |  1 |           pp512 |      2544.53 ± 75.10 |
llama-bench: benchmark 8/12: starting
llama-bench: benchmark 8/12: warmup generation run
llama-bench: benchmark 8/12: generation run 1/5
llama-bench: benchmark 8/12: generation run 2/5
llama-bench: benchmark 8/12: generation run 3/5
llama-bench: benchmark 8/12: generation run 4/5
llama-bench: benchmark 8/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |      512 |   q8_0 |   q8_0 |  1 |           tg128 |        112.25 ± 1.68 |
llama-bench: benchmark 9/12: starting
llama-bench: benchmark 9/12: warmup prompt run
llama-bench: benchmark 9/12: prompt run 1/5
llama-bench: benchmark 9/12: prompt run 2/5
llama-bench: benchmark 9/12: prompt run 3/5
llama-bench: benchmark 9/12: prompt run 4/5
llama-bench: benchmark 9/12: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |     1024 |   q8_0 |   q8_0 |  1 |           pp512 |      2531.40 ± 67.32 |
llama-bench: benchmark 10/12: starting
llama-bench: benchmark 10/12: warmup generation run
llama-bench: benchmark 10/12: generation run 1/5
llama-bench: benchmark 10/12: generation run 2/5
llama-bench: benchmark 10/12: generation run 3/5
llama-bench: benchmark 10/12: generation run 4/5
llama-bench: benchmark 10/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |     1024 |   q8_0 |   q8_0 |  1 |           tg128 |        112.27 ± 1.25 |
llama-bench: benchmark 11/12: starting
llama-bench: benchmark 11/12: warmup prompt run
llama-bench: benchmark 11/12: prompt run 1/5
llama-bench: benchmark 11/12: prompt run 2/5
llama-bench: benchmark 11/12: prompt run 3/5
llama-bench: benchmark 11/12: prompt run 4/5
llama-bench: benchmark 11/12: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |     2048 |   q8_0 |   q8_0 |  1 |           pp512 |      2503.40 ± 75.72 |
llama-bench: benchmark 12/12: starting
llama-bench: benchmark 12/12: warmup generation run
llama-bench: benchmark 12/12: generation run 1/5
llama-bench: benchmark 12/12: generation run 2/5
llama-bench: benchmark 12/12: generation run 3/5
llama-bench: benchmark 12/12: generation run 4/5
llama-bench: benchmark 12/12: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |    2048 |     2048 |   q8_0 |   q8_0 |  1 |           tg128 |        111.07 ± 1.30 |

build: 8974ec829 (8780)
