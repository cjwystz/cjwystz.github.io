---
layout: post
title: "910B3 推理研究的九条线索和实验结果"
date: 2026-10-11 00:00:00 +0800
categories: [inference, llm-serving]
tags: [ascend-910b3, sglang, hccl, prefix-cache, disaggregation, reproducibility]
description: 汇总计算干扰、KV 入站、真实 PD、hybrid 缓存和多模态方向的调研与实验，并记录通信执行路径候选从微基准交叉点到完整 TP2 对照的筛选过程。
toc:
  beginning: true
---

这轮推理系统研究沿着九条线索展开。最值得记录的现象是：**同一个通信模式在人工并发实验中可以改变优劣排序，到了完整模型里，支撑这个想法的并发条件却没有出现。** hybrid 缓存也有类似情况：部分前缀不能复用，看起来像一个调度机会；检查现有实现并调整 chunk 后，许多缺口已经被基线解决。

目前还没有验证成论文方案的 idea。但筛选范围已经更具体：不要求新范式，先找一个真实瓶颈，用小机制解决，再对比现有强基线。下面把做过什么、哪些结果能保留、下一步如何判断放在一起。

实验实际完成于 **2026 年 10 月 8 日至 9 日**，本文按系列顺序标为 10 月 11 日。主要硬件是 Ascend 910B3，单卡 64 GB HBM。完整服务实验主要使用 SGLang 0.5.18、CANN 9.0.0、TorchNPU 2.10；不同模型、软件栈和卡数之间不直接比较加速比。[上一篇](/blog/2026/ascend-910b3-inference-audit-negative-results/)详细记录了计时修正、单卡基线和 attention 原型，本篇补上跨节点、hybrid 与通信路径的后续结果。

## 已完成的实验和调研

按保存的结果核对，至少有 **35 组主运行具备完整 JSON 记录**。这是一个保守下限，早期大量 JSONL 微基准没有纳入；运行完成也不等于通过所有正确性检查。文献记录去重得到 **44 个 arXiv 论文 ID**，阅读以摘要、关键系统章节和实现为主，不能称为 44 篇全文精读。

| 方向 | 已完成的主要检查 | 当前判断 |
|---|---|---|
| 计时与计算干扰 | 校正 stream/event 计时，比较不同提交顺序 | 干扰仍存在，原 10.2 倍说法撤回，根因没有单独确定 |
| Vector attention 与核配额 | 数值检查、完整 0.6B/8B 模型、合法配额对照 | 原型未转化为完整模型收益 |
| 单卡 serving 基线 | 8 个配置，共 688 个计时请求 | 调优已有 chunk 即可解决大部分排队 |
| 跨节点 KV 入站 | 4 个主 fixture，共 200 个接收端 cases | 单次入站影响很小，持续入站有有限影响 |
| 真实 PD | 4 轮相同 trace，共 344 个计时请求 | 建立真实 KV 交接路径，未验证新调度收益 |
| TP2 与入站 | 原基线、到达率标定和入站对照，共 434 个计时请求 | batch 并发下有干扰，但尚未形成独特机制 |
| Hybrid 缓存 | 36 个固定配对 query、24 个增长分支 query，以及两类状态恢复 fixture | 固定配对和 HF 状态恢复通过，增长分支严格 ID 检查有分歧 |
| 多模态 | ChartQA 256 次预算评估、32 次原生服务评估；GUI 210 次窗口检查 | 有质量与成本差异，尚未验证可部署控制器 |
| 通信执行路径 | 270 个主对照 joint cases、30 个微基准 profile cases、150 个完整 TP2 计时请求 | 人工并发排序交叉成立，当前 dense TP2 不具备对应竞争条件 |

完整模型服务计时请求合计 **1,616 个**：688 单卡 + 344 真实 PD + 434 原 TP2/入站 + 150 新通信模式对照。warmup 和 smoke 不计入，hybrid 与多模态检查另列。重复请求不是独立工作负载，两个通信 rank 也只构成一个 joint trial；表中不同单位不能相加成“实验次数”。

## 先排除基线和测量造成的机会

修正计时后，合成 GEMM 与 softmax 共执行仍能让 decode event span 增长约 3 至 4 倍。但这只说明排队或执行干扰存在，不能只凭 Cube/Vector 的利用率就归因于 HBM。随后做的 vector attention 在完整模型里没有收益，缩减或预留核数也没有救回这个原型。

单卡 serving 更直接。相同 86 请求 trace 下，已有的非 mixed 配置把 chunk 从 512 调到 2048，满足 TTFT/平均 TPOT SLO 的请求从 8 个增加到 80 个。这个变化说明，拿未调优配置做基线，会把普通参数问题误认为新调度机会。这里仍是固定形状的随机 token 负载，不是语言质量评测，也不是多 trace 容量结论。

跨节点实验把“一次搬运”和“持续搬运”分开。一个 576 MiB 合成 BF16 buffer 入站后，在完整 Qwen3-8B 的十个固定 decode steps 上，平均影响不到 1%；连续 32 次、累计 18 GiB 入站时，B1 和 B8 的平均 step 延迟分别增加约 7.13% 和 3.97%。

<img src="/assets/img/blog/research-screening-20261009/ingress_slowdown.png" alt="一次与持续跨节点入站在完整模型固定步 graph 中的平均延迟影响" style="display:block;margin:1em auto;width:90%;max-width:100%;height:auto;" />

图里的接收 buffer 是合成负载，固定步 graph 也不是在线自回归 serving。真实 Qwen3-0.6B PD 则跑通了独立 prefill、KV 交接、decode 与 router；相同 trace 的四轮 SLO 达成数为 0、61、83、84/86。连续运行状态差异很大，不能把后几轮改善当作新方法加速。

TP2 serving 的 batch 入站配对中，中央 token 间隔平均增加约 4.39%。这是一个可以复查的干扰信号，但还没有证明现有通信策略解决不了它。因此，普通“KV 接收会干扰 decode”暂时降为低优先级。

## 通信路径在微基准里出现了排序交叉

这条候选的具体问题是：最快的独立通信路径，在同时运行计算时，是否仍然最快？如果计算负载会改变路径排序，固定模式可能不适合所有执行阶段。

[CANN 的 HcclCommConfig 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclcpp_07_0047.html)已有 communicator 粒度的展开模式选择。本轮只实测默认 mode 0 和 forced AIV mode 4；显式 HOST mode 1 没有运行。AIV 使用设备 Vector Core，存在算子与组网限制，文档还明确不支持多个 AIV communicator 并行。因此，固定模式本来就应当是调优基线，不能把找到一个参数开关当作新机制。

初筛使用同机两张卡、一个 HCCL communicator，运行 BF16 AllReduce；另一个 stream 上执行矩阵计算 graph 或 softmax graph。逐 shape 预热，随机 case 顺序，主机同步和 Gloo barrier 放在计时外。汇总时取同一个 trial 中两个 rank 的较大完成时间。

反转模式启动顺序、把重复数增加到十次，并让每次输入变化、输出先置为无效值后，16/32 MiB 的排序仍然交叉：

| 同时运行的计算 | 16 MiB 的 AIV 相对默认耗时变化 | 32 MiB 的变化 |
|---|---:|---:|
| 无 | −5.36% | −6.45% |
| 矩阵计算 | **+20.50%** | **+18.06%** |
| softmax | −6.00% | −6.02% |

负数表示更快。这支持了“独立通信速度不足以预测共执行速度”的现象，但方向也提醒我们：不能直接按最初的 Vector 竞争猜测解释成因。

首次 AIV 运行曾发生完整 payload 检查失败，原始失败保留并排除出性能对照。修订 fixture，让所有捕获 graph 和 buffer 存活到 communicator 销毁后，后续检查通过；这还不足以证明首次失败的根因。确认组每次使用新输入、poison 输出，避免把旧结果留在 buffer 中也误算通过。

另做的设备 profile，每个模式有 15 个 joint cases，通信与计算输出都通过检查。每个 rank 观察到 9 次 AllReduce、96 个 MatMul 和 96 个 Softmax；AIV 模式另有 9 个 AivKernel，默认模式没有。人工并发的设备时间线确实存在重叠。

<img src="/assets/img/blog/research-screening-20261009/comm_engine_device_overlap.png" alt="默认和 AIV 两种模式在人工计算通信并发条件下的设备时间线" style="display:block;margin:1em auto;width:95%;max-width:100%;height:auto;" />

各面板使用自己的时间原点，只展示选定 rank 和 repeat 的窗口。logical collective 的跨度可能包含等待；profile 用于确认路径和重叠，不把 profiler 开启后的耗时当作服务加速证据。

这条方向的近邻也需要认真比较。[HyperParallel-MoE](https://arxiv.org/html/2605.23764v1)已经在 Ascend MoE training 中联合调度 AIC/AIV 任务、通信和计算。它与这里的 dense inference 场景不同，但泛泛的“协调 Cube、Vector 和通信”显然不够成为贡献。

## 完整 TP2 对照让这条路线暂时停止

接下来在完整 Qwen3-8B 上做 **默认 → forced AIV → 默认复跑**。三次都是独立服务启动，同机两卡 TP2，page 128、token pool 16384、chunk 2048、max-running 8、context 8192、decode graph 开启、radix cache 关闭。只修改本次服务 TP communicator 的配置。

预先固定三种 workload：B1/输入512/输出256、B8/输入1024/输出256、B1/输入4096/输出32。每种先做三轮完整 warmup，再检查 32 个 greedy 输出 IDs；计时阶段关闭 logprob，每种五个 batch trials。设备 profile 在全部计时结束后另采。

三次启动的 32-ID smoke 都通过，计时请求长度也正确。不过，B8 的完整计时输出文本在默认模式自身的重复中已有差异；三次启动的文本检查分别为 false、true、false。这一组排除出性能结论。32-token smoke 通过，不能掩盖 256-token 计时输出的分歧；文本分歧也不能直接解释为语义质量下降。

其余两个 shape 的对照如下。中央 ITL 使用输出第 65 至 192 个 token 区间的单 token 增量，不把合并的 SSE 消息当作一个 token。

| 指标 | 默认首轮 | Forced AIV | 默认复跑 |
|---|---:|---:|---:|
| B1/512/256 中央 ITL | 12.155 ms | 11.846 ms | **11.161 ms** |
| B1/4096/32 整请求耗时 | 731.423 ms | **704.501 ms** | 731.373 ms |

<img src="/assets/img/blog/research-screening-20261009/comm_tp_fixed_mode_control.png" alt="完整 TP2 三次独立启动中通过文本检查的两种 workload 对照" style="display:block;margin:1em auto;width:90%;max-width:100%;height:auto;" />

柱形是五个 trials 的均值，点是各 trial。短输入里，AIV 相对首轮默认快 2.54%，相对复跑默认却慢 6.14%；默认自身漂移达 −8.18%，所以不成立稳定加速。长输入单请求的整请求耗时相对两个默认约低 3.7%，可保留为固定配置的局部信号；三次独立启动还不是置信区间，也不是动态方法验证。

更关键的是设备路径。两种模式、三种 shape、两个 rank 的十二个 profile 窗口中，**MatMul 与 logical collective 的时间区间交集全部为 0**。AIV 模式确实执行了 AivKernel，但当前完整 dense TP2 没有人工两 stream 实验中的矩阵计算竞争。

因此，暂时停止为这套配置设计“计算压力感知通信选路”控制器。这个判断只覆盖本次模型、shape 和执行配置，不外推 MoE、其他重叠执行引擎、跨节点 TP 或全部 attention/Vector kernel。微基准现象可以保留，应用它的真实条件还需要另外证明。

## Hybrid 缓存已经解决了哪些问题

原先考虑的是：KV 与 recurrent checkpoint 分布在不同位置时，最深可恢复前缀是否总是最便宜？更浅的本地状态加重算，是否可能胜过远端恢复？

这个问题周围已经有较强基线。[Marconi](https://arxiv.org/html/2411.19379v3)研究 hybrid 缓存的 admission 和 compute-aware eviction；[Sparse Prefix Caching](https://arxiv.org/html/2605.05219v1)研究稀疏 checkpoint 的位置选择；[SuffixReplay](https://arxiv.org/html/2609.33477v1)已有独立 anchor sidecar 与 fetch/replay 流水。[SGLang Unified Radix Cache](https://www.sglang.io/blog/unified-radix-cache)也按各组件共同接受的最深边界恢复。一个宽泛的“缓存、网络、重算联合决策”还需要更具体的未解决机制。

完整 Qwen3.5-0.8B 的原生检查里，prefill chunk 1024 时，共享512/768/1024个 token 的 query 实际命中为 0/0/1024。只把 chunk 调为256，命中就变成512/768/1024。两组共18 pairs、36个 query，32个输出 IDs 的配对检查都通过。

<img src="/assets/img/blog/research-screening-20261009/hybrid_prefix_readiness.png" alt="两种已有 prefill chunk 配置下不同共享前缀的实际缓存命中 token 数" style="display:block;margin:1em auto;width:85%;max-width:100%;height:auto;" />

图展示缓存可用性，不是新调度器加速比。首次新 shape 还有明显耗时离群点，不能把 producer 后的所有请求均值解释为稳态恢复成本。

状态恢复也跑通了两类 fixture。HF 完整模型提取的48个 cache tensors，经 CPU 保存、加载后回 NPU，逐元素一致；8步续算的 greedy IDs 相同、full-logit relative RMSE 均为0。跨节点 HCCL 搬运的13,483,219字节 snapshot 通过全 payload hash 检查，接收端续算也通过同样的8步检查。这里包含 CPU staging，HF 与原生引擎的 recurrent dtype 也不同，所以不是生产原生 connector 或跨引擎状态互换。

增长分支 trace 给出更细的结果。三次重复中，四个 query 的实际命中都为0/512/512/768，32-output ID 是否等于 cold reference 都是 true/true/false/true。第三个 query 从512 checkpoint 继续计算，在第2个输出 token 出现分歧；之后直接使用建立好的768 checkpoint 又匹配。

这说明引擎会补充新分支状态，首次部分 miss 不能当成永久损失。输出分歧还可能与随机 token prompt、计算分段和数值放大有关，不能据此断言 checkpoint 损坏。[LMCache 的 hybrid 文档](https://raw.githubusercontent.com/LMCache/LMCache/dev/docs/source/mp/hybrid_models.rst)也提醒，相关 GDN 缓存路径不能一概用逐 token 一致性判断质量。后续应在有意义文本上补充固定 continuation 的 full logits 和任务分数，再决定是否存在需要修复的具体问题。

## 多模态先测质量代价和复用上限

在真实 [ChartQA](https://github.com/vis-nlp/ChartQA) 的64题分层样本上，Qwen2.5-VL-3B 使用64/256/1024/4096最大视觉 token 预算，答对数为8/47/53/53；首步 wall time 中位数约240/294/389/387 ms。1024与4096有61题实际 grid 和输出相同，所以最大预算提高不等于实际视觉计算提高。

256到1024有9题改善，也有3题退化。事后挑每题最好的预算可得到56/64，但这是 oracle，不能当作一个可部署的预算策略。另用原生 SGLang 做16题、两档预算对照，256为12/16、1024为15/16；这建立了质量与耗时检查，没有验证新控制器。HF 与原生输入处理不同，也不做跨框架 identical-input 加速声明。

GUI 方向检查了70个真实截图转移、三个预算档，共210次窗口统计。它们只来自两个 task，无法代表广泛 agent 工作负载。Qwen2.5-VL 的早期局部 attention 层允许复用未变化窗口，但第一次全局混合后，变化通常会传播。Calc 样本在1024预算下，前七层 token 复用率中位数为73.81%，并不等于整个 encoder 快73.81%。首个原型的特征检查通过，graph capture 却失败，所以没有性能结论。

这一方向保留的具体问题是：变化检测、缓存管理、重算和回退全部算进去后，能否在真实连续画面上减少端到端首步耗时，同时守住质量？目前还没有通过这个检查。

## 下一项优先检查 PD 的驻留生命周期

比继续扩大“联合调度”的范围更有用的是，先画清每个请求在两端的实际内存占用。候选问题是：decode 提前预留目标 KV 页，而 prefill 或传输尚未完成时，这段占用是否足以让两端容量利用失衡？

只读当前实现后，必须区分三段时间：目标端在源端 prefill 时的提前预留、传输期间的双端副本，以及传输完成后的 decode 等待。**源端 KV 在 transfer 成功时释放，并不默认一直保留到 decode 接纳。** 因此，不能把所有 decode 排队都算成“双端驻留”。

下一组诊断要按 request ID 记录实际分配字节、预留与发布、prefill 完成、传输完成和 decode 接纳，先判断瓶颈是否存在。不同节点的时间不能未经校验就直接相减。[WAIT](https://arxiv.org/html/2504.11320v1)等内存约束调度是重要近邻，延迟预留与调优背压也必须作为强基线；若它们已消除损失，就停止这条候选。人为缩小小模型内存池制造出的容量问题不计为真实收益。

这仍是待测假设。一个范围较窄、机制清楚、有真实 workload 收益的结果值得继续；只有一般现象、静态参数改善或较弱基线上的漂亮倍数，还不足以立项。

## 数据和复算入口

[下载本轮精简证据附件](/assets/data/research-screening-20261009/evidence.zip)（约1.2 MB）。附件包含选定原始 JSON、失败样本、汇总、实验脚本和逐文件 SHA256 清单。解压后可运行：

```bash
python3 summarize_comm_engine_probe.py
python3 recompute_serving_timings.py
```

前者按两个 rank 的 joint trial 复算人工并发结果；后者从完整 TP2 原始请求复算计时和文本检查，不重做设备 profile 审计。[总盘点](/assets/data/research-screening-20261009/results/research_progress_inventory.json)、[完整 TP2 汇总](/assets/data/research-screening-20261009/results/comm_tp_serving_summary.json)、[hybrid 汇总](/assets/data/research-screening-20261009/results/hybrid_readiness_summary.json)也可直接读取。

数 GB 的完整设备轨迹和 CANN 原始二进制没有放进站点仓库，研究归档另行保留，公开 profile 汇总记录了所用 CSV 的 hash。模型权重与数据集也需要单独取得。此前单卡实验和 ChartQA 的原始记录继续保留在[上一篇的附件](/blog/2026/ascend-910b3-inference-audit-negative-results/)。
