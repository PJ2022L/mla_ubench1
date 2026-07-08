#!/usr/bin/env python3
"""组合模型：读各原子 log → 用重叠感知解析式(max/+/×)预测 T_fused → 对拍 e2e → η + 瓶颈。
SCAFFOLD。精确结构见 docs/05-sm90-sparse-decode-model.md。

单位统一为 cycle。每个原子 micro-bench 报告单次操作周期 T_x：
  T_wgmma_qk : 一条 wgmma m64n64k16      (a4)
  T_wgmma_pv : 一条 wgmma m64n256k16     (a5)
  T_softmax  : 一个 block 的 softmax     (a6)
  T_prod_block[cross|nocross] : producer 处理一个 64-token block（gather+dequant+dsm）(a1+a2+a3)
  T_A7_store, T_A8_combine    : epilogue (a7,a8)

模型（V3.2, per §6 of doc 05）:
  # 消费者关键路径 = WG0 链（WG1 remote-PV 被掩盖）
  T_consumer = QK_ISSUES*T_wgmma_qk + T_softmax + PV_ISSUES*T_wgmma_pv
  T_producer = T_prod_block
  T_block    = max(T_producer, T_consumer)          # 双缓冲重叠
  T_fused    = T_prologue + (n_block-1)*T_block + T_epilogue
"""
import argparse

# ---- 每 block 的 WGMMA 发射次数（源码核实，doc 05 §3）----
QK_ISSUES = 576 // 16   # = 36  (HEAD_DIM_K / 16)
PV_ISSUES = 64 // 16    # = 4   (TOPK_BLOCK_SIZE / 16)


def compose(T, n_block, split_kv=False):
    """T: dict of measured atom cycles. 返回预测与诊断。"""
    T_consumer = QK_ISSUES * T["T_wgmma_qk"] + T["T_softmax"] + PV_ISSUES * T["T_wgmma_pv"]
    T_producer = T["T_prod_block"]                       # 默认取 cross 版
    T_block = max(T_producer, T_consumer)

    T_prologue = T_producer                              # 首个 producer 无重叠对象
    T_epilogue = T["T_A7_store"] + (T["T_A8_combine"] if split_kv else 0.0)
    T_fused = T_prologue + (n_block - 1) * T_block + T_epilogue

    T_serial_upper = T_producer + T_consumer             # 零重叠上界（每 block）
    bottleneck = "producer(memory/dequant-bound)" if T_producer > T_consumer else "consumer(compute-bound)"
    return {
        "T_consumer": T_consumer, "T_producer": T_producer,
        "T_block": T_block, "T_fused": T_fused,
        "T_upper_per_block": T_serial_upper, "bottleneck": bottleneck,
    }


def crossover_gain(T):
    """DSM crossover 收益（doc 05 §5）：无 crossover producer 时间 − 有 crossover。"""
    if "T_prod_block_nocross" not in T or "T_prod_block" not in T:
        return None
    return T["T_prod_block_nocross"] - T["T_prod_block"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms-dir", default="../atoms")
    ap.add_argument("--e2e-log", default="../e2e/log")
    ap.add_argument("--n-block", type=int, default=32)   # topk/TOPK_BLOCK = 2048/64
    ap.add_argument("--split-kv", action="store_true")
    args = ap.parse_args()

    # TODO(impl):
    #   1) parse atoms/*/log → T = {T_wgmma_qk, T_wgmma_pv, T_softmax, T_prod_block[,_nocross], T_A7_store, T_A8_combine}
    #      注意各原子 REPEAT 换算：单次周期 = 实测总 cycles / REPEAT / (每次操作的指令数)。
    #   2) r = compose(T, args.n_block, args.split_kv)
    #   3) parse e2e/log → T_measured_cycles（用 getGPUClock 把 ms↔cycle）；η = r["T_fused"] / T_measured_cycles。
    #   4) 打印: 原子上限表 + T_producer/T_consumer + bottleneck + η + 上界；crossover_gain(T)。
    print(f"[TODO] compose (QK_ISSUES={QK_ISSUES}, PV_ISSUES={PV_ISSUES}, n_block={args.n_block}) -> vs e2e")


if __name__ == "__main__":
    main()
