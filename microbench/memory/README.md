# Memory benchmarks

只放以数据移动、cache 或 memory fabric 为主的原子：`ld/st.global`、TMA、shared-memory store、DSM 和带宽主导的 split reduction。固定为 `memory/<family>/<configuration>/`；格式转换和 softmax 归 `../compute/`。
