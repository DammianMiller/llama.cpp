# Phase B v2 — q8_0 MMA pre-dequant (stride fix)

A/B at d=32768. Previous baselines: f16/f16 = 108-109 t/s, q8_0/q8_0 vec = 91-95 t/s.

## With Phase B enabled

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  pp512 @ d32768 |      2155.24 ± 47.87 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         99.86 ± 0.25 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |    f16 |  1 |  pp512 @ d32768 |      2177.68 ± 53.02 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |    f16 |  1 |  tg128 @ d32768 |         96.84 ± 0.40 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |   q8_0 |  1 |  pp512 @ d32768 |      2142.00 ± 33.56 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |   q8_0 |  1 |  tg128 @ d32768 |         98.94 ± 0.73 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |    f16 |  1 |  pp512 @ d32768 |      2168.29 ± 39.50 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |    f16 |  1 |  tg128 @ d32768 |        116.03 ± 0.47 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  pp512 @ d32768 |      2132.45 ± 31.33 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         98.28 ± 0.45 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |    f16 |  1 |  pp512 @ d32768 |      2154.06 ± 31.95 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |    f16 |  1 |  tg128 @ d32768 |         96.47 ± 0.44 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |   q8_0 |  1 |  pp512 @ d32768 |      2199.80 ± 32.18 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |   q8_0 |  1 |  tg128 @ d32768 |        102.04 ± 0.74 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |    f16 |  1 |  pp512 @ d32768 |      2241.81 ± 32.50 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |    f16 |    f16 |  1 |  tg128 @ d32768 |        118.66 ± 0.52 |

build: 29247d088 (8885)
## Phase B disabled (control: vec kernel baseline)

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  pp512 @ d32768 |      2150.14 ± 29.43 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |        101.48 ± 0.93 |

build: 29247d088 (8885)
