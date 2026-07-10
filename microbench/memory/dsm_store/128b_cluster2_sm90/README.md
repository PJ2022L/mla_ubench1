# 128b_cluster2_sm90

- **Target**：`st.async.weak.shared::cluster` 128-bit peer-CTA store。
- **Geometry**：cluster `(2,1,1)`，128 producer threads，local/peer address 与 transaction barrier 成对。
- **Timed body**：peer mapping + async store + 必需完成等待；不含 global load/FP8 convert。
- **Metrics**：cycle/store、byte/clk/cluster、cycle/64-token producer block。
- **Consumers**：FlashMLA SM90 sparse FP8 decode cluster2 crossover。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 128-bit store 与 cluster size 2。
