# Blackwell Elementwise Throughput

HBM-resident fp32 elementwise benchmark for B200. The default operation sweep
matches the Trainium VectorEngine benchmark operations: `exp`, `tanh`, `rsqrt`,
`add`, `multiply`, and `relu`.

Each kernel reads one fp32 HBM input vector, applies one operation, and writes
one fp32 HBM output vector. CUDA events provide event-time throughput, while
`benchmark.py` also collects Nsight Compute DRAM and instruction metrics when
`ncu` is available.

```bash
python3 benchmark.py --overwrite -o elementwise_tput_results.csv
```

Useful reduced run:

```bash
python3 benchmark.py --ops exp,relu --elements 16777216 --iters 5 --overwrite
```

The benchmark has not been run on the Trainium host because that machine has no
NVIDIA GPU, `nvcc`, or `ncu`. Run it on a B200 host before using it as a
measured Blackwell source in cross-accelerator comparisons.
