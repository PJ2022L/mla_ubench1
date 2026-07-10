# Performance modeling — pending atom decomposition

模型必须从 prefill 的真实 pipeline 推导。仅在两个阶段可并行且同步关系允许时使用 `max`；矩阵 tile 循环使用 `N × T`；依赖链使用 `+`。
