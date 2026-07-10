# Sparse decode e2e benchmark

测量 V3.2 sparse FP8 public decode API 的稳态 GPU 区间：main sparse kernel 加 scheduler 实际需要的 combine。第一次调用生成的 metadata 和 testcase/FP8 quantization 不计时。

默认 production-like shape：`b=128,s_q=2,h_q=128,h_kv=1,s_k=32768,topk=2048,d_qk=576,d_v=512`。

```powershell
conda activate peijun
python benchmark.py --batch 128 --s-q 2 --s-k 32768 --topk 2048 --iters 100
```

脚本复用 `target/tests/lib.py` 的 V3.2 cache quantization 和 FLOPS/bytes 统计，所以还需要上游 test 依赖 `kernelkit`。输出 latency/TFLOPS/GB/s；main/combine 分解使用 Kineto/ncu。
