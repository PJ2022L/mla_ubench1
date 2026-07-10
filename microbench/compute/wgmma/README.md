# WGMMA family

层级固定为 `wgmma/<mM nN kK>_<dtype>_<operand-mode>_<arch>/`。family 共享 `benchmark.cu`；叶子 Makefile 只固定 MNK/dtype/mode，并在叶子 README 定义测量边界。

RS 与 SS 即使在同一配置中实现，也必须分开运行和报告。延时实验使用 accumulator 依赖链；吞吐实验使用多个独立 accumulator/足够 resident warpgroups。两者都必须明确 `commit_group`/`wait_group` 是否计入。
