# Dense decode e2e benchmark

测量公开 `flash_mla_with_kvcache` 的稳态 GPU 区间。API 每次都会 launch dense main 和 combine grid；split request 的 combine 做实际 reduction，no-split CTA 会提前 return。计时不包含第一次调用的 metadata 生成和 tensor 初始化。

默认 shape：`b=128,s_q=1,h_q=128,h_kv=1,d=576,d_v=512,s_k=4096,page=64`。建议扫描 `s_q∈{1,2}`、`s_k∈{4096,8192,16384,32768}`，并加入非 64 整倍数的 tail。`s_q=1` 时上游 C++ 强制关闭 causal，所以该 case 不应把 `--causal` 记作有效 variant。

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"
e2e_dir="$repo_root/operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e"

# CPU-only argument check; does not import torch or FlashMLA.
python "$e2e_dir/benchmark.py" --validate-only --batch 128 --s-q 1 --s-k 4097 --causal

# Run only in the FlashMLA environment on the remote H800 host.
run_id="$(date +%Y%m%d-%H%M%S)-$(hostname)"
run_dir="$e2e_dir/result/runs/$run_id"
mkdir -p "$run_dir"
/usr/bin/time -f 'command=%C\nwall_seconds=%e\nexit_status=%x' \
  python "$e2e_dir/benchmark.py" --batch 128 --s-q 1 --s-k 4096 --iters 100 \
  >"$run_dir/results.jsonl" 2>"$run_dir/run.log"
```

前置条件：在 H800/SM90a、CUDA 12.8+ 环境中，从仓库根目录执行 `git -C "$repo_root/operators/flash_mla/target" submodule update --init --recursive` 和 `python -m pip install -v "$repo_root/operators/flash_mla/target"`。

输出 JSON 包含平均 e2e latency、effective TFLOPS/GB/s、请求和实际生效的 causal 状态、page 数及完整 shape。这里的 FLOPS/bytes 是算法最小工作量，不含 scheduler metadata、split partial traffic 和 combine traffic，不能当作实测硬件流量。若需要拆分 main/combine 时间，使用上游 Kineto test 或 ncu，不能把单 main-kernel 时间称为 e2e。

## H800 Results

| Accepted run | Full shape | Correctness | Latency (ms) | Effective TFLOPS / GB/s | Main / combine boundary |
|---|---|---|---:|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整结果保存在本目录 `result/runs/<run_id>/`：`results.jsonl` 只保留正式
correctness/latency 结果，`run.log` 保留 args、命令、wall time 和错误。accepted
run 必须链接到本表，并记录 requested/effective causal、page/tail、实际
`num_splits`、output/LSE correctness，以及 profiler 确认的 main/combine 边界。
