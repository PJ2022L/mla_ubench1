# Compute benchmarks

只放 Tensor Core、格式转换、SFU 和数值归约原子。固定为 `compute/<family>/<configuration>/`；例如 `wgmma/m64n64k16_bf16_rs_ss_sm90/`。WGMMA 配置名必须写明 M/N/K、dtype、operand mode 和架构。
