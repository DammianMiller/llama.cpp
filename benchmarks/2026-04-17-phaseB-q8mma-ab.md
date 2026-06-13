# Phase B — q8_0 MMA pre-dequant A/B

Hot-GPU interleaved A/B at d=32768. Reference: f16/f16 at 108-109 t/s, q8_0/q8_0 vec-path at 91-95 t/s.

## With Phase B (GGML_CUDA_FA_QUANT_MMA enabled, default)

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
/home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/ggml/src/ggml-cuda/ggml-cuda.cu:98: CUDA error
[New LWP 1028092]
[New LWP 1028090]
[New LWP 1028089]
[New LWP 1028088]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.ubuntu.com>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
__syscall_cancel_arch () at ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S:56
warning: 56	../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S: No such file or directory
#0  __syscall_cancel_arch () at ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S:56
56	in ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S
#1  0x00007fc8d32a013c in __internal_syscall_cancel (a1=<optimized out>, a2=<optimized out>, a3=<optimized out>, a4=<optimized out>, a5=0, a6=0, nr=61) at ./nptl/cancellation.c:49
warning: 49	./nptl/cancellation.c: No such file or directory
#2  __syscall_cancel (a1=<optimized out>, a2=<optimized out>, a3=<optimized out>, a4=<optimized out>, a5=a5@entry=0, a6=a6@entry=0, nr=61) at ./nptl/cancellation.c:75
75	in ./nptl/cancellation.c
#3  0x00007fc8d331ca0f in __GI___wait4 (pid=<optimized out>, stat_loc=<optimized out>, options=<optimized out>, usage=<optimized out>) at ../sysdeps/unix/sysv/linux/wait4.c:30
warning: 30	../sysdeps/unix/sysv/linux/wait4.c: No such file or directory
#4  0x00007fc8d432ddd3 in ggml_print_backtrace () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libggml-base.so.0
#5  0x00007fc8d432df86 in ggml_abort () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libggml-base.so.0
#6  0x00007fc8d1f3bf47 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libggml-cuda.so.0
#7  0x00007fc8d1f3d588 in ggml_backend_cuda_synchronize(ggml_backend*) () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libggml-cuda.so.0
#8  0x00007fc8d434b33d in ggml_backend_sched_graph_compute_async () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libggml-base.so.0
#9  0x00007fc8d44bf390 in llama_context::graph_compute(ggml_cgraph*, bool) () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libllama.so.0
#10 0x00007fc8d44c1d53 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libllama.so.0
#11 0x00007fc8d44c88cf in llama_context::decode(llama_batch const&) () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libllama.so.0
#12 0x00007fc8d44c9f12 in llama_decode () from /home/cogtek/llama.cpp/.worktrees/upgrade-b8740-turbo/build-q8mma/bin/libllama.so.0
#13 0x000055d0f3ee37cb in test_prompt(llama_context*, int, int, int) ()
#14 0x000055d0f3ee0093 in main ()
[Inferior 1 (process 1028086) detached]
## Same but with Phase B disabled (fall back to vec kernel)

ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24123 MiB
| model                          |       size |     params | backend    | ngl | n_batch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  pp512 @ d32768 |      2057.22 ± 73.08 |
| qwen35moe 35B.A3B IQ4_XS - 4.25 bpw |  16.50 GiB |    34.66 B | CUDA       |  99 |     512 |   q8_0 |   q8_0 |  1 |  tg128 @ d32768 |         93.89 ± 0.69 |

build: 29247d088 (8885)
