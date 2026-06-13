# llama.cpp Service Config Profiles

Verified production configurations for the UAP llama-server service on RTX 3090 24GB.

## Profile: Qwen3.5-35B-A3B (Default)

**Saved as**: `/home/cogtek/.config/uap/llama-server.env.35B-A3B-default`

### Config
```bash
LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2/build-pq/bin/llama-server
LLAMA_MODEL=/home/cogtek/Downloads/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf

LLAMA_HOST=0.0.0.0
LLAMA_PORT=8080
LLAMA_CTX_SIZE=262144
LLAMA_THREADS=32
LLAMA_CACHE_TYPE_K=q8_0
LLAMA_CACHE_TYPE_V=q8_0
LLAMA_PARALLEL=2
LLAMA_GPU_LAYERS=99
LLAMA_BATCH_SIZE=512
LLAMA_UBATCH_SIZE=512
LLAMA_ENABLE_SPEC_DECODING=false
LLAMA_EXTRA_ARGS=--metrics --kv-unified
```

### Characteristics
| Property | Value |
|----------|-------|
| Architecture | Hybrid MoE (3B active of 34.66B) |
| Model size | 16.28 GiB |
| VRAM used | ~22 GiB (90%) |
| KV buffer | 2,720 MiB |
| **Measured throughput** | **95-112 tok/s** |
| Per-slot context | 262,144 tokens |
| Slots | 2 concurrent |

### Why Spec OFF
Hybrid draft model (Qwen3.5-0.8B) overhead exceeds savings on this MoE model (already fast at 3B active params). Measured: spec on = -27% for tool calls.

### Tag
`planarquant-spec-deferred-v1` (branch: `feature/planarquant-kv`)

---

## Profile: Qwen3.5-27B (Dense, Spec Enabled)

**Saved as**: `/home/cogtek/.config/uap/llama-server.env.27B-spec`

### Config
```bash
LLAMA_BIN=/home/cogtek/llama.cpp/.worktrees/turboquant-cuda-v2/build-pq/bin/llama-server
LLAMA_MODEL=/home/cogtek/Downloads/Qwen3.5-27B-IQ4_XS.gguf

LLAMA_HOST=0.0.0.0
LLAMA_PORT=8080
LLAMA_CTX_SIZE=262144
LLAMA_THREADS=32
LLAMA_CACHE_TYPE_K=q8_0
LLAMA_CACHE_TYPE_V=q8_0
LLAMA_PARALLEL=2
LLAMA_GPU_LAYERS=99
LLAMA_BATCH_SIZE=512
LLAMA_UBATCH_SIZE=512
LLAMA_ENABLE_SPEC_DECODING=true
LLAMA_SPEC_TYPE=ngram-cache
LLAMA_DRAFT_MAX=2
LLAMA_DRAFT_MIN=1
LLAMA_DRAFT_P_MIN=0.5
LLAMA_EXTRA_ARGS=--metrics --kv-unified
```

### Characteristics
| Property | Value |
|----------|-------|
| Architecture | Hybrid dense (all 26.9B active) |
| Model size | 13.94 GiB |
| **Expected throughput** | **35-42 tok/s** (spec can help) |

### Why Spec ON
Dense 27B is baseline-slow (33 tok/s), so draft overhead is a smaller relative cost. In measured tests: +10-24% with draft-max=2.

---

## Switching Between Profiles

```bash
# To 35B-A3B default:
cp /home/cogtek/.config/uap/llama-server.env.35B-A3B-default /home/cogtek/.config/uap/llama-server.env
systemctl --user restart uap-llama-server

# To 27B with spec:
cp /home/cogtek/.config/uap/llama-server.env.27B-spec /home/cogtek/.config/uap/llama-server.env
systemctl --user restart uap-llama-server
```
