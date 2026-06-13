# KV dtype A/B confirm — hot GPU, interleaved

43, 1260 MHz

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 49152 | int dot: 0 | matrix cores: NV_coopmat2
llama-bench: benchmark 1/8: starting
llama-bench: benchmark 1/8: warmup generation run
llama-bench: benchmark 1/8: depth run 1/5
llama-bench: benchmark 1/8: generation run 1/5
llama-bench: benchmark 1/8: depth run 2/5 (cached)
llama-bench: benchmark 1/8: generation run 2/5
llama-bench: benchmark 1/8: depth run 3/5 (cached)
llama-bench: benchmark 1/8: generation run 3/5
llama-bench: benchmark 1/8: depth run 4/5 (cached)
llama-bench: benchmark 1/8: generation run 4/5
llama-bench: benchmark 1/8: depth run 5/5 (cached)
llama-bench: benchmark 1/8: generation run 5/5
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         91.53 ± 1.88 |
llama-bench: benchmark 2/8: starting
llama-bench: benchmark 2/8: warmup generation run
llama-bench: benchmark 2/8: depth run 1/5
llama-bench: benchmark 2/8: generation run 1/5
llama-bench: benchmark 2/8: depth run 2/5 (cached)
llama-bench: benchmark 2/8: generation run 2/5
llama-bench: benchmark 2/8: depth run 3/5 (cached)
llama-bench: benchmark 2/8: generation run 3/5
llama-bench: benchmark 2/8: depth run 4/5 (cached)
llama-bench: benchmark 2/8: generation run 4/5
llama-bench: benchmark 2/8: depth run 5/5 (cached)
llama-bench: benchmark 2/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |    f16 |  1 |  tg128 @ d32768 |         91.55 ± 0.54 |
llama-bench: benchmark 3/8: starting
llama-bench: benchmark 3/8: warmup generation run
llama-bench: benchmark 3/8: depth run 1/5
llama-bench: benchmark 3/8: generation run 1/5
llama-bench: benchmark 3/8: depth run 2/5 (cached)
llama-bench: benchmark 3/8: generation run 2/5
llama-bench: benchmark 3/8: depth run 3/5 (cached)
llama-bench: benchmark 3/8: generation run 3/5
llama-bench: benchmark 3/8: depth run 4/5 (cached)
llama-bench: benchmark 3/8: generation run 4/5
llama-bench: benchmark 3/8: depth run 5/5 (cached)
llama-bench: benchmark 3/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q8_0 |  1 |  tg128 @ d32768 |         93.36 ± 0.83 |
llama-bench: benchmark 4/8: starting
llama-bench: benchmark 4/8: warmup generation run
llama-bench: benchmark 4/8: depth run 1/5
llama-bench: benchmark 4/8: generation run 1/5
llama-bench: benchmark 4/8: depth run 2/5 (cached)
llama-bench: benchmark 4/8: generation run 2/5
llama-bench: benchmark 4/8: depth run 3/5 (cached)
llama-bench: benchmark 4/8: generation run 3/5
llama-bench: benchmark 4/8: depth run 4/5 (cached)
llama-bench: benchmark 4/8: generation run 4/5
llama-bench: benchmark 4/8: depth run 5/5 (cached)
llama-bench: benchmark 4/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |    f16 |  1 |  tg128 @ d32768 |        108.30 ± 0.60 |
llama-bench: benchmark 5/8: starting
llama-bench: benchmark 5/8: warmup generation run
llama-bench: benchmark 5/8: depth run 1/5
llama-bench: benchmark 5/8: generation run 1/5
llama-bench: benchmark 5/8: depth run 2/5 (cached)
llama-bench: benchmark 5/8: generation run 2/5
llama-bench: benchmark 5/8: depth run 3/5 (cached)
llama-bench: benchmark 5/8: generation run 3/5
llama-bench: benchmark 5/8: depth run 4/5 (cached)
llama-bench: benchmark 5/8: generation run 4/5
llama-bench: benchmark 5/8: depth run 5/5 (cached)
llama-bench: benchmark 5/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         94.76 ± 1.13 |
llama-bench: benchmark 6/8: starting
llama-bench: benchmark 6/8: warmup generation run
llama-bench: benchmark 6/8: depth run 1/5
llama-bench: benchmark 6/8: generation run 1/5
llama-bench: benchmark 6/8: depth run 2/5 (cached)
llama-bench: benchmark 6/8: generation run 2/5
llama-bench: benchmark 6/8: depth run 3/5 (cached)
llama-bench: benchmark 6/8: generation run 3/5
llama-bench: benchmark 6/8: depth run 4/5 (cached)
llama-bench: benchmark 6/8: generation run 4/5
llama-bench: benchmark 6/8: depth run 5/5 (cached)
llama-bench: benchmark 6/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |   q8_0 |    f16 |  1 |  tg128 @ d32768 |         91.49 ± 1.21 |
llama-bench: benchmark 7/8: starting
llama-bench: benchmark 7/8: warmup generation run
llama-bench: benchmark 7/8: depth run 1/5
llama-bench: benchmark 7/8: generation run 1/5
llama-bench: benchmark 7/8: depth run 2/5 (cached)
llama-bench: benchmark 7/8: generation run 2/5
llama-bench: benchmark 7/8: depth run 3/5 (cached)
llama-bench: benchmark 7/8: generation run 3/5
llama-bench: benchmark 7/8: depth run 4/5 (cached)
llama-bench: benchmark 7/8: generation run 4/5
llama-bench: benchmark 7/8: depth run 5/5 (cached)
llama-bench: benchmark 7/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |   q8_0 |  1 |  tg128 @ d32768 |         93.09 ± 1.44 |
llama-bench: benchmark 8/8: starting
llama-bench: benchmark 8/8: warmup generation run
llama-bench: benchmark 8/8: depth run 1/5
llama-bench: benchmark 8/8: generation run 1/5
llama-bench: benchmark 8/8: depth run 2/5 (cached)
llama-bench: benchmark 8/8: generation run 2/5
llama-bench: benchmark 8/8: depth run 3/5 (cached)
llama-bench: benchmark 8/8: generation run 3/5
llama-bench: benchmark 8/8: depth run 4/5 (cached)
llama-bench: benchmark 8/8: generation run 4/5
llama-bench: benchmark 8/8: depth run 5/5 (cached)
llama-bench: benchmark 8/8: generation run 5/5
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA,Vulkan |  99 |     512 |    f16 |    f16 |  1 |  tg128 @ d32768 |        109.17 ± 0.95 |

build: 8974ec829 (8780)
