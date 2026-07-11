# Sparse decode e2e benchmark

测量 V3.2 sparse FP8 public decode API 的稳态 GPU 区间：main sparse kernel 加 scheduler 实际需要的 combine。第一次调用生成的 metadata 和 testcase/FP8 quantization 不计时。

默认 production-like shape：`b=128,s_q=2,h_q=128,h_kv=1,s_k=32768,topk=2048,d_qk=576,d_v=512`。

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"
e2e_dir="$repo_root/operators/flash_mla/paths/sparse_decode_fp8_sm90_v32_mqa_h128_cluster2/e2e"

# CPU-only argument check; does not import torch, FlashMLA, kernelkit, lib, or ref.
python "$e2e_dir/benchmark.py" --validate-only --batch 128 --s-q 2 --s-k 32768 --topk 2048

# Run only on the remote H800 host after installing target/ and its test dependencies.
python "$repo_root/tools/result_tool.py" run --result-dir "$e2e_dir/result" --kind e2e -- \
  python "$e2e_dir/benchmark.py" --batch 128 --s-q 2 --s-k 32768 --topk 2048 --iters 100
```

脚本复用 `target/tests/lib.py` 的 V3.2 cache quantization 和 FLOPS/bytes 统计，所以还需要上游 test 依赖 `kernelkit`。输出 latency/TFLOPS/GB/s；main/combine 分解使用 Kineto/ncu。

脚本会显式设置 CUDA 默认 device/dtype，与上游 testcase runner 一致。`--check` 会运行 PyTorch reference，只应配合显著缩小的 batch/topk shape 使用。CUDA-event 区间包含 main + PDL combine；metadata 初始化与 FP8 quantization 在第一次调用完成，不进入稳态计时。

## H800 Results

| Accepted run | Full shape | Correctness | Latency (ms) | Effective TFLOPS / GB/s | Main / combine boundary |
|---|---|---|---:|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整结果保存在本目录 `result/runs/<run_id>/`，包含 `metadata.json`、`result.jsonl` 和 `run.log`；`result/summary.csv` 由这些不可变 runs 重建。accepted run 必须链接到本表，并记录完整 sparse shape、reference correctness、实际 scheduler split/partition，以及 profiler 确认的 main/combine 边界。
