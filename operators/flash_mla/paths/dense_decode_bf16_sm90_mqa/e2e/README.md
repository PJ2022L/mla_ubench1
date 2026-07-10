# Dense decode e2e benchmark

测量公开 `flash_mla_with_kvcache` 的稳态 GPU 区间，包含 dense main kernel 和按 scheduler 需要启动的 combine；不包含第一次调用的 metadata 生成和 tensor 初始化。

默认 shape：`b=128,s_q=1,h_q=128,h_kv=1,d=576,d_v=512,s_k=4096,page=64`。建议扫描 `s_q∈{1,2}`、`s_k∈{4096,8192,16384,32768}`、causal on/off。

```powershell
conda activate peijun
python benchmark.py --batch 128 --s-q 1 --s-k 4096 --iters 100
```

前置条件：在 H800/SM90a、CUDA 12.8+ 环境进入 `../../../target/`，执行 `git submodule update --init --recursive` 和 `pip install -v .`。

输出 JSON 至少包含平均 e2e latency、TFLOPS、GB/s 和完整 shape。若需要拆分 main/combine 时间，使用上游 Kineto test 或 ncu，不能把单 main-kernel 时间称为 e2e。
