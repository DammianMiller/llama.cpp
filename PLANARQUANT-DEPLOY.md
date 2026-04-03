# PlanarQuant Production Deployment Guide

## Tags

| Component | Tag | Commit | Location |
|---|---|---|---|
| **llama.cpp** | `planarquant-v1.0-working` | `0179871c4` | `/home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2` |
| **UAP proxy** | `llama-planarquant-v1.0` | `12f402ed` | `/home/cogtek/dev/miller-tech/universal-agent-protocol` |

Branch: `feature/planarquant-kv` (on turboquant-cuda-v2 worktree, based on `fix/hybrid-spec-perf`)

## Model

```
File:     Qwen3.5-35B-A3B-UD-IQ4_XS.gguf
Size:     17 GB
SHA256:   745def087c9ec34d12a94eb9b31eb52dba958b36def510b7e8f6f2e6bf1be123
Location: /home/cogtek/Downloads/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf
Source:   unsloth/Qwen3.5-35B-A3B-GGUF (HuggingFace)
Type:     Hybrid attention + Mamba SSM (qwen35moe architecture)
```

## System

```
GPU:      NVIDIA GeForce RTX 3090, 24576 MiB VRAM
Driver:   590.48.01
CUDA:     12.4 (compiler 34097967)
CPU:      AMD Ryzen 9 5950X 16-Core
OS:       Linux 6.19.6-x64v3-xanmod1
GCC:      15.2.0
```

## Build Steps

```bash
cd /home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2
git checkout feature/planarquant-kv  # or: git checkout planarquant-v1.0-working

mkdir -p build-pq && cd build-pq
cmake .. \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

make -j$(nproc) llama-server llama-bench llama-speculative-simple
```

**Critical flags:**
- `GGML_CUDA_FA_ALL_QUANTS=ON` — required for PQ flash attention CUDA kernels
- `CMAKE_CUDA_ARCHITECTURES=86` — SM_86 for RTX 3090 (Ampere)

## Server Configuration

**Config file:** `/home/cogtek/.config/uap/llama-server.env`

```env
LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2/build-pq/bin/llama-server
LLAMA_MODEL=/home/cogtek/Downloads/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf

LLAMA_HOST=0.0.0.0
LLAMA_PORT=8080
LLAMA_CTX_SIZE=131072
LLAMA_THREADS=32
LLAMA_CACHE_TYPE_K=q8_0
LLAMA_CACHE_TYPE_V=q4_0
LLAMA_PARALLEL=1
LLAMA_GPU_LAYERS=99
LLAMA_BATCH_SIZE=512
LLAMA_UBATCH_SIZE=512
LLAMA_ENABLE_SPEC_DECODING=true
LLAMA_SPEC_TYPE=ngram-cache
LLAMA_DRAFT_MAX=8
LLAMA_DRAFT_MIN=3
LLAMA_DRAFT_P_MIN=0.75
LLAMA_LOG_FILE=/home/cogtek/llama.cpp/llama-server.log
LLAMA_CHAT_TEMPLATE_FILE=/home/cogtek/dev/miller-tech/universal-agent-protocol/tools/agents/config/chat_template.jinja
LLAMA_EXTRA_ARGS=--metrics
```

**Resulting CLI:**
```bash
llama-server \
    --model Qwen3.5-35B-A3B-UD-IQ4_XS.gguf \
    --host 0.0.0.0 --port 8080 \
    --threads 32 --ctx-size 131072 \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    --gpu-layers 99 --flash-attn on \
    --batch-size 512 --ubatch-size 512 \
    --parallel 1 --no-context-shift \
    --n-predict 81920 --repeat-penalty 1.05 --temp 0.3 \
    --spec-type ngram-cache --draft-max 8 --draft-min 3 --draft-p-min 0.75 \
    --metrics \
    --chat-template-file chat_template.jinja \
    --log-file llama-server.log
```

## Systemd Service

**Unit file:** `/home/cogtek/.config/systemd/user/uap-llama-server.service`

```ini
[Unit]
Description=llama.cpp server (continuity profile)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/cogtek/dev/miller-tech/universal-agent-protocol
EnvironmentFile=/home/cogtek/.config/uap/llama-server.env
ExecStart=/home/cogtek/dev/miller-tech/universal-agent-protocol/scripts/run-llama-server-continuity.sh
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=default.target
```

**Commands:**
```bash
systemctl --user restart uap-llama-server.service
systemctl --user status uap-llama-server.service
journalctl --user -u uap-llama-server.service -f
```

## Memory Layout (131k context)

```
KV cache:   1040 MiB total
  K (q8_0):  680 MiB  (131072 cells × 10 attn layers × 512 dim)
  V (q4_0):  360 MiB  (131072 cells × 10 attn layers × 512 dim)
Model:     ~16.3 GiB  (IQ4_XS weights on CUDA)
Compute:   ~489 MiB   (graph working memory)
Recurrent: ~19.3 MiB  (Mamba SSM state, 24 layers)
Total:     ~18.0 GiB  (of 24 GiB available)
```

## Verified Performance (2026-04-03)

### Generation throughput
| Metric | Value |
|---|---|
| Average generation | 118.2 tok/s |
| Best (ngram hit) | 218.6 tok/s |
| Worst (cache miss) | 85.8 tok/s |
| Prompt throughput | 2066 tok/s |

### Ngram-cache speculative decoding
| Metric | Value |
|---|---|
| Acceptance (warm, similar topic) | 89-92% |
| Acceptance (cold, novel content) | 2-44% |
| Draft tokens per step | max 8 |
| Checkpoint interval | 8192 tokens |

### 132k context benchmark (llama-bench)
| V-Cache | pp512 (tok/s) | tg128 (tok/s) | pp131k+tg32 (tok/s) |
|---|---|---|---|
| F16 | 2863 | 115.7 | 1754 |
| q8_0 | 2704 | 113.5 | 1740 |
| q4_0 | 2591 | 110.3 | 1694 |
| pq4_0 | 2519 | 109.9 | 319* |
| pq3_0 | 2298 | 107.4 | 131* |

*PQ types use vec kernel (no MMA tensor core support yet), causing slower prompt processing.

## Available KV Cache Types

All types from the turboquant branch plus PlanarQuant:

| Type | Bits | Method | MMA Support |
|---|---|---|---|
| f16 | 16 | Native | Yes |
| q8_0 | 8 | Absmax scalar | Yes |
| q4_0 | 4 | Absmax scalar | Yes |
| turbo2_0 | 2 | WHT + PolarQuant | Yes |
| turbo3_0 | 3 | WHT + PolarQuant + QJL | Yes |
| turbo4_0 | 4 | WHT + PolarQuant + QJL | Yes |
| **pq4_0** | **4.125** | **Givens rotation + Lloyd-Max** | **Vec only** |
| **pq3_0** | **3.125** | **Givens rotation + Lloyd-Max** | **Vec only** |

## Recreating From Scratch

```bash
# 1. Clone llama.cpp and checkout the turboquant base
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# Apply the turboquant-cuda-v2 patches (from the worktree)
git checkout -b feature/planarquant-kv
# Then cherry-pick or apply the PlanarQuant commit:
git cherry-pick 0179871c4

# 2. Build
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc) llama-server llama-bench

# 3. Download model
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
    Qwen3.5-35B-A3B-UD-IQ4_XS.gguf --local-dir ./models/

# 4. Run
./bin/llama-server \
    --model ./models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf \
    --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q4_0 \
    --gpu-layers 99 --flash-attn on --parallel 1 \
    --spec-type ngram-cache --draft-max 8 --draft-min 3 --draft-p-min 0.75

# 5. Verify
curl http://localhost:8080/health
curl http://localhost:8080/metrics | grep predicted_tokens_seconds
```

## Known Limitations

1. **PQ types require GGML_CUDA_FA_ALL_QUANTS** — without it, PQ flash attention kernels aren't compiled
2. **PQ uses vec kernel only** — MMA tensor core support not implemented, causing 5x slower prompt processing at 132k
3. **Hybrid model spec decoding** — uses checkpoint rollback (turboquant branch fix), not partial seq_rm
4. **Deferred K-cache quantization** — not implemented; requires KV cache architecture changes
5. **PQ3_0 3-bit packing** — bitstream packing adds ~3% overhead vs PQ4_0 nibble packing
