| 版本              | 核心架构/路线                  | 主要变化                                                 | 对应论文/报告                                                |
| ----------------- | ------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| DeepSeek LLM      | Dense decoder-only Transformer | LLaMA-like 架构，2T tokens，SFT+DPO                      | *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* |
| DeepSeek-Coder    | Dense code model               | 代码语料、FIM、16K 上下文                                | *DeepSeek-Coder: When the Large Language Model Meets Programming* |
| DeepSeekMoE       | MoE 架构预研                   | fine-grained experts + shared experts                    | *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models* |
| DeepSeek-V2       | MoE + MLA                      | 236B 总参数，21B 激活，128K，上下文成本大幅降低          | *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model* |
| DeepSeek-Coder-V2 | V2 架构代码化                  | 基于 V2 继续训练，338 种语言，128K                       | *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence* |
| DeepSeek-V3       | 大规模 MoE + MLA               | 671B 总参数，37B 激活，MTP，aux-loss-free load balancing | *DeepSeek-V3 Technical Report*                               |
| DeepSeek-R1       | V3 架构 + RL 推理              | 重点是 RL 激发推理，不是底层结构大改                     | *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* |
| DeepSeek-V3.1     | V3 系列后训练增强              | Think/Non-Think hybrid inference，Agent 能力增强         | 官方 release/model card                                      |
| DeepSeek-V3.2     | DSA 稀疏注意力                 | 长上下文效率、Agent 任务合成、工具使用推理               | *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models* |
| DeepSeek-V4       | 百万上下文 MoE                 | CSA+HCA、mHC、Muon、1M context，1.6T/49B                 | *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* |