# Param sweep A — KV cache dtype at d=32768

Model: Qwen3.6-35B-A3B-UD-IQ4_XS | ngl 99, fa 1, b/ub 512, t 16

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 49152 | int dot: 0 | matrix cores: NV_coopmat2
llama-bench: benchmark 1/18: starting
llama-bench: benchmark 1/18: warmup prompt run
llama-bench: benchmark 1/18: depth run 1/5
llama-bench: benchmark 1/18: prompt run 1/5
llama-bench: benchmark 1/18: depth run 2/5 (cached)
llama-bench: benchmark 1/18: prompt run 2/5
llama-bench: benchmark 1/18: depth run 3/5 (cached)
llama-bench: benchmark 1/18: prompt run 3/5
llama-bench: benchmark 1/18: depth run 4/5 (cached)
llama-bench: benchmark 1/18: prompt run 4/5
llama-bench: benchmark 1/18: depth run 5/5 (cached)
llama-bench: benchmark 1/18: prompt run 5/5
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |    f16 |  1 |  pp512 @ d32768 |      2083.21 ± 29.87 |
llama-bench: benchmark 2/18: starting
llama-bench: benchmark 2/18: warmup generation run
llama-bench: benchmark 2/18: depth run 1/5 (cached)
llama-bench: benchmark 2/18: generation run 1/5
llama-bench: benchmark 2/18: depth run 2/5 (cached)
llama-bench: benchmark 2/18: generation run 2/5
llama-bench: benchmark 2/18: depth run 3/5 (cached)
llama-bench: benchmark 2/18: generation run 3/5
llama-bench: benchmark 2/18: depth run 4/5 (cached)
llama-bench: benchmark 2/18: generation run 4/5
llama-bench: benchmark 2/18: depth run 5/5 (cached)
llama-bench: benchmark 2/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |    f16 |  1 |  tg128 @ d32768 |        110.62 ± 0.70 |
llama-bench: benchmark 3/18: starting
llama-bench: benchmark 3/18: warmup prompt run
llama-bench: benchmark 3/18: depth run 1/5
llama-bench: benchmark 3/18: prompt run 1/5
llama-bench: benchmark 3/18: depth run 2/5 (cached)
llama-bench: benchmark 3/18: prompt run 2/5
llama-bench: benchmark 3/18: depth run 3/5 (cached)
llama-bench: benchmark 3/18: prompt run 3/5
llama-bench: benchmark 3/18: depth run 4/5 (cached)
llama-bench: benchmark 3/18: prompt run 4/5
llama-bench: benchmark 3/18: depth run 5/5 (cached)
llama-bench: benchmark 3/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q8_0 |  1 |  pp512 @ d32768 |       2081.10 ± 8.38 |
llama-bench: benchmark 4/18: starting
llama-bench: benchmark 4/18: warmup generation run
llama-bench: benchmark 4/18: depth run 1/5 (cached)
llama-bench: benchmark 4/18: generation run 1/5
llama-bench: benchmark 4/18: depth run 2/5 (cached)
llama-bench: benchmark 4/18: generation run 2/5
llama-bench: benchmark 4/18: depth run 3/5 (cached)
llama-bench: benchmark 4/18: generation run 3/5
llama-bench: benchmark 4/18: depth run 4/5 (cached)
llama-bench: benchmark 4/18: generation run 4/5
llama-bench: benchmark 4/18: depth run 5/5 (cached)
llama-bench: benchmark 4/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q8_0 |  1 |  tg128 @ d32768 |         94.67 ± 0.59 |
llama-bench: benchmark 5/18: starting
llama-bench: benchmark 5/18: warmup prompt run
llama-bench: benchmark 5/18: depth run 1/5
llama-bench: benchmark 5/18: prompt run 1/5
llama-bench: benchmark 5/18: depth run 2/5 (cached)
llama-bench: benchmark 5/18: prompt run 2/5
llama-bench: benchmark 5/18: depth run 3/5 (cached)
llama-bench: benchmark 5/18: prompt run 3/5
llama-bench: benchmark 5/18: depth run 4/5 (cached)
llama-bench: benchmark 5/18: prompt run 4/5
llama-bench: benchmark 5/18: depth run 5/5 (cached)
llama-bench: benchmark 5/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q4_0 |  1 |  pp512 @ d32768 |      2103.57 ± 38.23 |
llama-bench: benchmark 6/18: starting
llama-bench: benchmark 6/18: warmup generation run
llama-bench: benchmark 6/18: depth run 1/5 (cached)
llama-bench: benchmark 6/18: generation run 1/5
llama-bench: benchmark 6/18: depth run 2/5 (cached)
llama-bench: benchmark 6/18: generation run 2/5
llama-bench: benchmark 6/18: depth run 3/5 (cached)
llama-bench: benchmark 6/18: generation run 3/5
llama-bench: benchmark 6/18: depth run 4/5 (cached)
llama-bench: benchmark 6/18: generation run 4/5
llama-bench: benchmark 6/18: depth run 5/5 (cached)
llama-bench: benchmark 6/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q4_0 |  1 |  tg128 @ d32768 |         94.22 ± 1.23 |
llama-bench: benchmark 7/18: starting
llama-bench: benchmark 7/18: warmup prompt run
llama-bench: benchmark 7/18: depth run 1/5
llama-bench: benchmark 7/18: prompt run 1/5
llama-bench: benchmark 7/18: depth run 2/5 (cached)
llama-bench: benchmark 7/18: prompt run 2/5
llama-bench: benchmark 7/18: depth run 3/5 (cached)
llama-bench: benchmark 7/18: prompt run 3/5
llama-bench: benchmark 7/18: depth run 4/5 (cached)
llama-bench: benchmark 7/18: prompt run 4/5
llama-bench: benchmark 7/18: depth run 5/5 (cached)
llama-bench: benchmark 7/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |    f16 |  1 |  pp512 @ d32768 |      2015.58 ± 31.96 |
llama-bench: benchmark 8/18: starting
llama-bench: benchmark 8/18: warmup generation run
llama-bench: benchmark 8/18: depth run 1/5 (cached)
llama-bench: benchmark 8/18: generation run 1/5
llama-bench: benchmark 8/18: depth run 2/5 (cached)
llama-bench: benchmark 8/18: generation run 2/5
llama-bench: benchmark 8/18: depth run 3/5 (cached)
llama-bench: benchmark 8/18: generation run 3/5
llama-bench: benchmark 8/18: depth run 4/5 (cached)
llama-bench: benchmark 8/18: generation run 4/5
llama-bench: benchmark 8/18: depth run 5/5 (cached)
llama-bench: benchmark 8/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |    f16 |  1 |  tg128 @ d32768 |         89.41 ± 0.34 |
llama-bench: benchmark 9/18: starting
llama-bench: benchmark 9/18: warmup prompt run
llama-bench: benchmark 9/18: depth run 1/5
llama-bench: benchmark 9/18: prompt run 1/5
llama-bench: benchmark 9/18: depth run 2/5 (cached)
llama-bench: benchmark 9/18: prompt run 2/5
llama-bench: benchmark 9/18: depth run 3/5 (cached)
llama-bench: benchmark 9/18: prompt run 3/5
llama-bench: benchmark 9/18: depth run 4/5 (cached)
llama-bench: benchmark 9/18: prompt run 4/5
llama-bench: benchmark 9/18: depth run 5/5 (cached)
llama-bench: benchmark 9/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q8_0 |  1 |  pp512 @ d32768 |      1996.26 ± 32.44 |
llama-bench: benchmark 10/18: starting
llama-bench: benchmark 10/18: warmup generation run
llama-bench: benchmark 10/18: depth run 1/5 (cached)
llama-bench: benchmark 10/18: generation run 1/5
llama-bench: benchmark 10/18: depth run 2/5 (cached)
llama-bench: benchmark 10/18: generation run 2/5
llama-bench: benchmark 10/18: depth run 3/5 (cached)
llama-bench: benchmark 10/18: generation run 3/5
llama-bench: benchmark 10/18: depth run 4/5 (cached)
llama-bench: benchmark 10/18: generation run 4/5
llama-bench: benchmark 10/18: depth run 5/5 (cached)
llama-bench: benchmark 10/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         91.21 ± 1.39 |
llama-bench: benchmark 11/18: starting
llama-bench: benchmark 11/18: warmup prompt run
llama-bench: benchmark 11/18: depth run 1/5
llama-bench: benchmark 11/18: prompt run 1/5
llama-bench: benchmark 11/18: depth run 2/5 (cached)
llama-bench: benchmark 11/18: prompt run 2/5
llama-bench: benchmark 11/18: depth run 3/5 (cached)
llama-bench: benchmark 11/18: prompt run 3/5
llama-bench: benchmark 11/18: depth run 4/5 (cached)
llama-bench: benchmark 11/18: prompt run 4/5
llama-bench: benchmark 11/18: depth run 5/5 (cached)
llama-bench: benchmark 11/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q4_0 |  1 |  pp512 @ d32768 |      2001.29 ± 42.14 |
llama-bench: benchmark 12/18: starting
llama-bench: benchmark 12/18: warmup generation run
llama-bench: benchmark 12/18: depth run 1/5 (cached)
llama-bench: benchmark 12/18: generation run 1/5
llama-bench: benchmark 12/18: depth run 2/5 (cached)
llama-bench: benchmark 12/18: generation run 2/5
llama-bench: benchmark 12/18: depth run 3/5 (cached)
llama-bench: benchmark 12/18: generation run 3/5
llama-bench: benchmark 12/18: depth run 4/5 (cached)
llama-bench: benchmark 12/18: generation run 4/5
llama-bench: benchmark 12/18: depth run 5/5 (cached)
llama-bench: benchmark 12/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q4_0 |  1 |  tg128 @ d32768 |         87.73 ± 0.60 |
llama-bench: benchmark 13/18: starting
llama-bench: benchmark 13/18: warmup prompt run
llama-bench: benchmark 13/18: depth run 1/5
llama-bench: benchmark 13/18: prompt run 1/5
llama-bench: benchmark 13/18: depth run 2/5 (cached)
llama-bench: benchmark 13/18: prompt run 2/5
llama-bench: benchmark 13/18: depth run 3/5 (cached)
llama-bench: benchmark 13/18: prompt run 3/5
llama-bench: benchmark 13/18: depth run 4/5 (cached)
llama-bench: benchmark 13/18: prompt run 4/5
llama-bench: benchmark 13/18: depth run 5/5 (cached)
llama-bench: benchmark 13/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |    f16 |  1 |  pp512 @ d32768 |      1946.36 ± 13.46 |
llama-bench: benchmark 14/18: starting
llama-bench: benchmark 14/18: warmup generation run
llama-bench: benchmark 14/18: depth run 1/5 (cached)
llama-bench: benchmark 14/18: generation run 1/5
llama-bench: benchmark 14/18: depth run 2/5 (cached)
llama-bench: benchmark 14/18: generation run 2/5
llama-bench: benchmark 14/18: depth run 3/5 (cached)
llama-bench: benchmark 14/18: generation run 3/5
llama-bench: benchmark 14/18: depth run 4/5 (cached)
llama-bench: benchmark 14/18: generation run 4/5
llama-bench: benchmark 14/18: depth run 5/5 (cached)
llama-bench: benchmark 14/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |    f16 |  1 |  tg128 @ d32768 |         92.95 ± 0.84 |
llama-bench: benchmark 15/18: starting
llama-bench: benchmark 15/18: warmup prompt run
llama-bench: benchmark 15/18: depth run 1/5
llama-bench: benchmark 15/18: prompt run 1/5
llama-bench: benchmark 15/18: depth run 2/5 (cached)
llama-bench: benchmark 15/18: prompt run 2/5
llama-bench: benchmark 15/18: depth run 3/5 (cached)
llama-bench: benchmark 15/18: prompt run 3/5
llama-bench: benchmark 15/18: depth run 4/5 (cached)
llama-bench: benchmark 15/18: prompt run 4/5
llama-bench: benchmark 15/18: depth run 5/5 (cached)
llama-bench: benchmark 15/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |   q8_0 |  1 |  pp512 @ d32768 |      2006.03 ± 57.78 |
llama-bench: benchmark 16/18: starting
llama-bench: benchmark 16/18: warmup generation run
llama-bench: benchmark 16/18: depth run 1/5 (cached)
llama-bench: benchmark 16/18: generation run 1/5
llama-bench: benchmark 16/18: depth run 2/5 (cached)
llama-bench: benchmark 16/18: generation run 2/5
llama-bench: benchmark 16/18: depth run 3/5 (cached)
llama-bench: benchmark 16/18: generation run 3/5
llama-bench: benchmark 16/18: depth run 4/5 (cached)
llama-bench: benchmark 16/18: generation run 4/5
llama-bench: benchmark 16/18: depth run 5/5 (cached)
llama-bench: benchmark 16/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |   q8_0 |  1 |  tg128 @ d32768 |         90.60 ± 0.54 |
llama-bench: benchmark 17/18: starting
llama-bench: benchmark 17/18: warmup prompt run
llama-bench: benchmark 17/18: depth run 1/5
llama-bench: benchmark 17/18: prompt run 1/5
llama-bench: benchmark 17/18: depth run 2/5 (cached)
llama-bench: benchmark 17/18: prompt run 2/5
llama-bench: benchmark 17/18: depth run 3/5 (cached)
llama-bench: benchmark 17/18: prompt run 3/5
llama-bench: benchmark 17/18: depth run 4/5 (cached)
llama-bench: benchmark 17/18: prompt run 4/5
llama-bench: benchmark 17/18: depth run 5/5 (cached)
llama-bench: benchmark 17/18: prompt run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |   q4_0 |  1 |  pp512 @ d32768 |      2023.99 ± 27.90 |
llama-bench: benchmark 18/18: starting
llama-bench: benchmark 18/18: warmup generation run
llama-bench: benchmark 18/18: depth run 1/5 (cached)
llama-bench: benchmark 18/18: generation run 1/5
llama-bench: benchmark 18/18: depth run 2/5 (cached)
llama-bench: benchmark 18/18: generation run 2/5
llama-bench: benchmark 18/18: depth run 3/5 (cached)
llama-bench: benchmark 18/18: generation run 3/5
llama-bench: benchmark 18/18: depth run 4/5 (cached)
llama-bench: benchmark 18/18: generation run 4/5
llama-bench: benchmark 18/18: depth run 5/5 (cached)
llama-bench: benchmark 18/18: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q4_0 |   q4_0 |  1 |  tg128 @ d32768 |         86.73 ± 1.26 |

build: 8974ec829 (8780)
