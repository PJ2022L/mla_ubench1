# Sparse prefill e2e benchmark

测量 public `flash_mla_sparse_fwd` 的完整 forward latency。默认 V3.2 performance shape：`s_q=4096,s_kv=8192,topk=2048,h_q=128,h_kv=1,d_qk=576,d_v=512`。

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"
e2e_dir="$repo_root/operators/flash_mla/paths/sparse_prefill_bf16_sm90_mqa/e2e"

# CPU-only argument check; OOB sparse indices make topk > s_kv a valid test shape.
python "$e2e_dir/benchmark.py" --validate-only --s-q 2 --s-kv 256 --topk 512

# Run only on the remote H800 host after installing target/ and its test dependencies.
run_id="$(date +%Y%m%d-%H%M%S)-$(hostname)"
run_dir="$e2e_dir/result/runs/$run_id"
mkdir -p "$run_dir"
/usr/bin/time -f 'command=%C\nwall_seconds=%e\nexit_status=%x' \
  python "$e2e_dir/benchmark.py" \
    --s-q 4096 --s-kv 8192 --topk 2048 --iters 50 \
  >"$run_dir/results.jsonl" 2>"$run_dir/run.log"
```

数据/indices 生成不计时。脚本复用上游 `target/tests/lib.py` 以保证布局和 FLOPS/bytes 口径一致。

脚本会显式设置 CUDA 默认 device/dtype，与上游 testcase runner 一致。`--check` 会运行 PyTorch reference，只应配合小 shape（例如 `--s-q 2 --s-kv 256 --topk 128`）使用。`gbps` 沿用上游的 logical/unique-byte 口径，不等于两个 64-head CTA 实际发出的 `cp.async` 总字节。

## H800 Results

| Accepted run | Full shape | Correctness | Latency (ms) | Effective TFLOPS / GB/s | Kernel boundary |
|---|---|---|---:|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整结果保存在本目录 `result/runs/<run_id>/`：`results.jsonl` 只保留正式结果，
`run.log` 保留 args、命令、wall time 和错误。accepted run 必须链接到本表，
并记录完整 sparse shape、reference correctness 和 profiler 确认的 kernel 边界。
此 path 没有 split-KV combine，不得套用 decode 的 main/combine 字段。
