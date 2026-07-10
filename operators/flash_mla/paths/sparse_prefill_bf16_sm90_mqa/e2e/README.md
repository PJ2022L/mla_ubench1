# Sparse prefill e2e benchmark

测量 public `flash_mla_sparse_fwd` 的完整 forward latency。默认 V3.2 performance shape：`s_q=4096,s_kv=8192,topk=2048,h_q=128,h_kv=1,d_qk=576,d_v=512`。

```powershell
conda activate peijun
python benchmark.py --s-q 4096 --s-kv 8192 --topk 2048 --iters 50
```

数据/indices 生成不计时。脚本复用上游 `target/tests/lib.py` 以保证布局和 FLOPS/bytes 口径一致。
