---
layout: post
title: "910B3 推理实验复核：从干扰微基准到完整模型的负结果"
date: 2026-10-10 00:00:00 +0800
categories: [inference, llm-serving]
tags: [ascend-910b3, profiling, sglang, attention, reproducibility, blog]
description: 修正双 stream 计时、重放相同请求 trace、调优 SGLang 基线，并验证 vector attention 与 KV 接收。干扰存在，但微基准优势没有转化为 Qwen3 完整模型收益。
toc:
  beginning: true
---

前两篇分别记录了 PD 配比和算子干扰。这次继续往下做，得到的主要结果却是：**原来的证据需要收紧，一个看起来不错的 attention 原型没有通过完整模型验证。** 把这些负结果记下来，比继续围绕一个漂亮倍数设计论文更有用。

本文实验实际完成于 **2026 年 10 月 8 日**，按博客系列顺序标注为 10 月 10 日。硬件是单机 Ascend 910B3，单卡 64GB HBM。serving 使用官方 SGLang 0.5.18 / CANN 9.0.0 镜像；旧形状计时复核和部分微基准使用 torch_npu 2.7，graph 与完整模型对照使用 2.10。不同软件栈的绝对时间不直接横向比较。

## 先校正前两篇的结论

上一篇的“decode 慢 10.2 倍”来自这样的流程：先向一个 stream 提交大量 prefill，再向另一个 stream 提交 decode，最后等待整个设备同步。这个终点还会等尚未完成的 prefill，**不能把全部等待时间除以 decode 步数，当作 decode 自身延迟。**

前一篇“3P:4D 吞吐高 11%”也只能保留为那次测试的观测。复核客户端后发现，到达间隔和 prompt 随机生成共用随机状态，相同 offered rate 下并不是同一条请求 trace；高档负载两组请求数约为 788 和 706。SSE 消息数也需要与实际生成 token 数区分。因此，这组数字不足以证明容量提高了 11%。新的容量比较应使用固定到达时间、固定输入和准确 token 计数重新跑。

还有一个归因要撤回：“prefill 用 Cube、decode 用 MTE，所以干扰只能来自内存”。原先 prefill 的 MTE2 ratio 本身就约为 0.87；利用率画像也不能排除调度和其他资源因素。NPU 的细分计数器很有用，但 GPU 同样有 tensor、内存和缓存等计数器，不能说 GPU 只能看一个 SM 占用率。

## 修正计时后，干扰仍然存在

新的脚本在 decode stream 上记录起止 NPU event，等待 decode 的结束 event 后再记完成时间；整个设备同步放到后面，用于清理并记录全部任务时间。另外加入串行、prefill 先提交、decode 先提交和交替提交，随机顺序重复五次。

下面仍是 **GEMM + softmax 合成负载**，不是完整 KV attention，也不是在线服务。表中 event span 包含 stream 执行和排队间隙，不等于各 kernel 时间之和。

| 提交方式 | decode event span / step（ms） | 相对单独执行 |
|---|---:|---:|
| decode 单独执行 | 0.227 | 1.00× |
| 先完成 prefill，再执行 decode | 0.227 | 1.00× |
| prefill 先提交，共执行 | 1.045 | 4.61× |
| decode 先提交，共执行 | 0.730 | 3.22× |
| 交替提交 | 1.047 | 4.62× |

修正后仍有干扰，不过不能再沿用原来的 10.2×。

另一个 fused attention + GEMM 实验中，profiler 时间线里，与 GEMM 重叠的 20 个 attention kernel 平均约 647.8µs，其他 60 个约 143.0µs。这说明执行时间增长不全是全设备同步造成的测量假象；**它仍没有单独确定 HBM、L2 或调度队列中的哪一项是根因。**

## 先调优已有 serving，再谈新方案

这次客户端把请求到达和 prompt 生成的随机数源分开，提前生成固定 trace。所有配置重放相同的 86 个请求：20 秒到达窗口，平均 4 req/s，输入为 512 / 4096 token 两种长度，输出固定为 128 token。使用随机 token IDs 测 shape workload，不代表真实语言任务。

服务关闭 radix cache，token pool 为 32768，max-running 为 32，context 上限 8192，graph 保持开启。客户端通过 cumulative completion_tokens 的增量记录真实 token，不能用 SSE 消息数代替。所有表内成功运行都完成了 86 个请求，输出长度准确。

预先指定请求级 SLO：**TTFT ≤ 1 秒，TPOT ≤ 50ms**。TPOT 是每个请求首末 token 间的平均间隔；逐 token ITL 另算，不能用平均 TPOT 替代每个 token 的 deadline。

| attention 路径 / batching | chunk | TTFT p50 / p99（s） | TPOT p99（ms） | 满足 SLO | 输出 tok/s |
|---|---:|---:|---:|---:|---:|
| 默认，非 mixed | 512 | 7.786 / 12.503 | 30.38 | 8 / 86 | 333.11 |
| 默认，非 mixed | 2048 | 0.150 / 1.287 | 16.88 | **80 / 86** | **535.36** |
| 默认，mixed | 2048 | 19.086 / 22.030 | 79.41 | 0 / 86 | 286.06 |
| FIA，非 mixed | 2048 | 0.627 / 2.986 | 37.93 | 53 / 86 | 469.42 |
| FIA，修复后的 mixed | 2048 | 3.286 / 6.155 | 37.14 | 3 / 86 | 502.22 |

吞吐包含请求到达窗口结束后的排空时间。最好的已有配置只是把非 mixed 的 chunk 从 512 调到 2048，就消除了大部分排队。它的逐 token ITL p99 仍为 74.24ms，因此“80 个请求满足平均 TPOT SLO”不表示每个 token 间隔都低于 50ms。

<img src="/assets/img/blog/audit_serving.png" alt="固定 trace 下已有 serving 配置的请求 SLO 达成数" style="display:block;margin:1em auto;max-width:100%;height:auto;" />

这张表是单条 trace 的诊断结果，还不是经过多条 held-out trace 和重复试验的容量结论。它已经足以说明：**不能把未调优基线的排队当作新方法的收益。**

### 为公平对照修复一个 FIA mixed 问题

启用 ASCEND_USE_FIA 后，KV pool 返回 token-major 视图 `[tokens, 1, heads, dim]`。mixed 路径却直接把第二维读成 block_size，得到 1，导致执行失败。

修复是在 mixed 路径使用 allocator 已有的 page_size 和 tensor 元素数恢复 page-major view。修改只发生在独立实验容器中，不改共享镜像。随机打乱物理页的 causal attention 测试与 FP32 dense 参考对照通过，最大绝对误差约 0.00172；这属于抽样算子正确性检查，不是模型质量评测。

两种服务路径实际 page_size 都是 128。此前猜测“默认 page_size 是 1”并不成立，问题来自视图解释。修复用于建立可运行的对照，不作为论文贡献。

## C/V 配额有合法域，不能任意切

torch_npu 提供 stream 的 Cube / Vector 数量限制，但这个接口不保证两条 stream 获得完全互斥的物理核心。

更具体的限制来自算子 tiling：在测试的普通 GQA FusedInferAttentionScore 路径中，4C:32V 失败；8C:8V 也失败，报错提示 1:1 只适用于指定 MLA 情形。当前这些 GQA 形状使用的合法比例是 **1C:2V**。

这不能外推为所有 Ascend attention 的统一限制。通用 FusionAttention 上接受的配额，也不能直接搬到生产 IFA 上。

对完整模型还要再检查一次：prefill 不只有 GEMM，还有 PFA。纯 GEMM 接受的 P16C8V 配额用于完整 prefill 后会报错，必须改成合法比例。整段 decode 限核也会影响 projection 和 FFN，不能把“attention 被保护了”直接理解成“完整 decode 更快”。

## vector attention 的亮点止于微基准

我实现了一个 exact decode attention 原型：QK 点积、online softmax 和 PV 都用向量归约，先 split，再 merge。profiler 确认 partial / merge 为 AI_VECTOR_CORE，而 vendor IFA 为 MIX_AIC。

在 B1、KV 长度 512、32 query heads / 8 KV heads、head_dim 128 的 FP16 graph 对照中，100 个 decode attention 与 100 个 4096² GEMM 共执行：

| 方案 | 共执行 attention 平均事件时间（ms） | 全部任务完成时间（ms） |
|---|---:|---:|
| vendor，不限核 | 0.4791 | 51.45 |
| vendor，D4C8V / P16C8V | 0.0779 | 64.30 |
| vector split4 | 0.0805 | 55.13 |

vector 在这个固定工作量里提供了一个 Pareto 点：接近限核方案的 attention 延迟，同时更早完成全部任务。这里的 P16C8V 只作用于 GEMM 竞争负载，所以可以执行；它不是完整 prefill 的合法配置。全任务时间也不是在线 serving 吞吐。

换成 B8 / KV 长度 8192，vector split16 已经约 2.425ms，vendor 共执行约 0.686ms，明显退化。换成 BF16 和包含 gate/up、SiLU、乘积、down projection 的完整 FFN 竞争负载后，差距也缩小：vendor attention 约 0.0988ms，vector 约 0.0844ms，全任务约 99.70 / 97.73ms。

### 完整训练模型的固定步验证

接下来加载真实 Qwen3-0.6B 和 Qwen3-8B 权重。每次 decode graph 从同一份真实 prefix KV 计算同一个 next-token logits，和完整 512-token prefill graph 共执行。decode batch=1，KV 长度 512；每个试验提交 10 个 decode graph 和 3 个 prefill graph，随机顺序重复三次。

这是**包含所有 transformer 层和 lm_head 的固定步 fixture**，不是完整自回归生成，也不是调优后的 SGLang 服务。权重使用 ND 格式以避开这套 HF graph 路径中 legacy GatherV2 的 capture 错误；绝对速度不用于比较 serving 框架。

| 模型 / 方案 | decode 单独执行（ms） | prefill 先提交时 decode 平均事件时间（ms） |
|---|---:|---:|
| Qwen3-0.6B，vendor | 5.829 | **8.700** |
| Qwen3-0.6B，vector | 5.997 | 8.872 |
| Qwen3-8B，vendor | 20.532 | **35.650** |
| Qwen3-8B，vector | 21.370 | 36.479 |
| Qwen3-8B，整段 D4C8V / P16C32V | 54.700 | 58.887 |

<img src="/assets/img/blog/audit_full_model.png" alt="完整训练模型固定步 fixture 中 vendor 与 vector 路径的共执行延迟" style="display:block;margin:1em auto;max-width:100%;height:auto;" />

预设输出检查门槛是相对 logits RMSE ≤ 2%，且 greedy token 相同。8B 的 vector / quota 相对误差分别约 0.619% / 1.343%，通过；0.6B 的 vector 约 1.132%，通过；0.6B quota 虽然 greedy 相同，但相对误差约 2.178%，超过门槛，排除性能比较。这个有限的 logits 检查不等同于长期生成和任务质量评测。

结果很直接：**当前只替换 decode attention 的实现没有完整模型收益，淘汰。** 微基准的互补性没有自然延伸到整个模型，不能把它包装成一篇已经有性能结果的论文。

## 模型规模的 KV 接收，当前也没有巨大干扰

最后检查 KV 接收是否会明显拖慢完整 decode。使用 8B 的真实配置：36 层、8 KV heads、head_dim 128、BF16、4096-token KV，总量为 576MiB。比较 pinned-host H2D、两张卡间整体 P2P 和逐层 P2P，每组五次随机顺序重复。

接收目标与活动请求的 prefix KV 分开，避免边写边读改变 decode 结果。数据规模来自模型配置，拷贝内容是受控常量并抽样验证；这是 copy 路径实验，不是线上 PD connector，也不是跨节点 RDMA。

| 接收方式 | 接收完成时间（ms） | decode 首步：单独 / 接收先提交（ms） | 十步平均：单独 / 接收先提交（ms） |
|---|---:|---:|---:|
| H2D | 27.26 | 20.492 / 21.167 | 20.481 / 20.568 |
| 整体 P2P | 29.08 | 20.502 / 21.823 | 20.472 / 20.644 |
| 逐层 P2P | 29.30 | 20.514 / 21.838 | 20.493 / 20.628 |

首步增加约 3.3%–6.5%，十步平均增加不足 1%。目前这组形状不支持“KV 接收普遍严重伤害 decode”的核心假设；持续并发接收、跨节点 RDMA 和长上下文 batch 尚需另测。

## 新增：真实图表问答的质量与耗时诊断

随后下载真实 Qwen2.5-VL-3B-Instruct 权重和 [ChartQA](https://github.com/vis-nlp/ChartQA) 的 2500 题 test 数据。在 human/augmented 两部分各随机取 32 题，固定种子 20261008，比较四档最大视觉 token。greedy 生成、自然 EOS、最多 32 个新 token；使用 [ChartQA relaxed accuracy](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/chartqa/utils.py) 的 5% 数值容差及非数值 exact match。

这是单请求 HF / torch_npu 2.10 / Transformers 5.12.1 的诊断，不是优化 serving 的吞吐结果。首步 wall time 包含 CPU 图像处理、输入搬运和第一次模型 forward，另保留视觉 event span；event span 含主机提交空隙，不等于纯芯片计算时间。模型首次 warmup 单独记录并排除汇总。

| 最大视觉 token | 实际视觉 token 中位数 | 答对题数 | 首步 wall time 中位数（ms） |
|---|---:|---:|---:|
| 64 | 54 | 8/64（12.5%） | 239.65 |
| 256 | 247 | 47/64（73.4%） | 293.74 |
| 1024 | 580 | 53/64（82.8%） | 389.48 |
| 4096 | 580 | 53/64（82.8%） | 387.17 |

压低分辨率确实能降低首步成本，但在这组样本上会严重损失答案质量。1024 与 4096 有 61/64 题得到相同 grid 和输出：这里调的是上限，图片到原生尺寸后不会继续放大，不能把这两档解释为实际计算量四倍却没有收益。

质量也不是逐题单调：256→1024 有 9 题由错变对，3 题由对变错。事后从所有预算选正确答案的 oracle 为 56/64，只用于观察选择空间；它读取了答案，**不是可部署方法的成绩**。样本仅 64 题，没有训练或验证任何预算控制器，不能当作全数据集准确率或新方法收益。分辨率控制已有 [ResAdapt](https://arxiv.org/html/2603.28610)、[SmartVL](https://arxiv.org/html/2607.20357) 等邻接工作，下一步先校准优化服务端，再找额外机制。

[逐题输出与时间记录](/assets/data/inference-audit-20261008/chartqa_budget64.jsonl)、[汇总](/assets/data/inference-audit-20261008/chartqa_budget64_summary.json)、[样本编号和数据 SHA256](/assets/data/inference-audit-20261008/chartqa_budget64.manifest.json)、[脚本](/assets/data/inference-audit-20261008/chartqa_budget_probe.py)。记录按原数据集行号关联，不重复分发图片和题目。

## 接下来怎样筛方向

一般的资源感知混部并不是空白。[xLLM](https://yangtonghome.github.io/uploads/xLLM.pdf) 已讨论 C/V 配额和算子重叠，[FlexNPU](https://arxiv.org/abs/2606.04415) 已有 phase-aware NPU 虚拟化与 PD 混部，[XY-Serve](https://arxiv.org/abs/2412.18106) 已有 NPU mixed attention。[REEF 扩展工作](https://doi.org/10.1145/3768622) 也包含面向共执行的 multi-version kernel 思想。它们需要进入相关工作和对照，不能只因为换了国产硬件就称首次。

下一轮把多模态作为候选方向：多页文档、长视频和同一视觉输入上的多问题请求，可能有 text-only trace 不会暴露的编码、缓存和精度约束。这里先保留研究问题，不把数据集换一个名字就认定为创新。新方法仍要通过机制、准确率、强基线和 held-out workload 四道检查。

今天最有用的产出是把证据分清：干扰确实存在；之前两个醒目的倍数不足以支撑原来的结论；已有调参能解决不少排队；vector 原型和大规模 KV 接收假设都没有通过当前收益门槛。

## 实验附件

[下载结果摘要与逐次记录](/assets/data/inference-audit-20261008/manifest.json)。附件保留软件与工作量说明，公开记录去除了内部模型绝对路径。结果文件中的 event span、whole time 和 serving 指标属于不同测量范围，不能混用。

- [修正计时的五次重复记录](/assets/data/inference-audit-20261008/mixed_audit_200.jsonl)
- [八组成功 serving 配置摘要](/assets/data/inference-audit-20261008/serving_summary.json)
- [Qwen3-0.6B 完整模型固定步记录](/assets/data/inference-audit-20261008/model_step_qwen06.jsonl)
- [Qwen3-8B 完整模型固定步记录](/assets/data/inference-audit-20261008/model_step_qwen8.jsonl)
- [KV 接收五次重复摘要](/assets/data/inference-audit-20261008/model_kv_transfer_summary.json)
- [完整模型测试脚本](/assets/data/inference-audit-20261008/model_step_comparison.py) 与 [vector 原型](/assets/data/inference-audit-20261008/vector_decode.py)
