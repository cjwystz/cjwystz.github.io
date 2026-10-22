---
layout: post
title: "基础知识巩固 02：KV 显存、连续批处理与推理延迟"
date: 2026-10-23 09:00:00 +0800
categories: [foundations, llm-serving]
tags: [kv-cache, gqa, paged-attention, continuous-batching, latency, learning-notes]
series: llm-foundations
description: 用不等长请求、分页和调度的手算例子，理解 KV 占用如何增长，以及 TTFT、TPOT 和吞吐为什么需要一起看。
math: true
toc:
  beginning: true
---

[第一篇](/blog/2026/llm-foundations-01-architecture-to-inference/)把从输入到输出的计算链路接了起来。这篇继续记几个服务系统中的基础问题：不同请求的 KV 应该怎样相加？分页到底省了什么？为什么连续批处理仍然会排队？延迟指标又该从哪里开始计时？

下面以普通全注意力、GQA、单条生成序列为例，暂不考虑投机解码、前缀共享和缓存卸载。数字均为**教学设定和算术结果，不是硬件性能实测**。目标是先把账算清楚，再去读具体框架的实现。

系列入口：[基础知识巩固](/blog/llm-foundations/)

## 1. 先从三个不同长度的请求开始

假设某个模型有32层，每层8个 KV head，每个 head 的 K、V 维度均为128，缓存使用 BF16，每个元素占2字节。

一个 token 在所有层的 KV 大小为：

$$
m_{token}=2\times32\times8\times128\times2
=131072\ \mathrm{bytes}=128\ \mathrm{KiB}
$$

最前面的2表示 K 和 V 两份。这里用的是 **KV head 数**，不是 Query head 数；GQA 中多个 Query head 共享一组 K/V，不应为每组 Query 重复计入缓存。

现在三条请求当前已缓存的长度分别如下：

| 请求 | 已缓存 token 数 | 有效 KV 数据 |
| --- | ---: | ---: |
| A | 4096 | 512 MiB |
| B | 2048 | 256 MiB |
| C | 1024 | 128 MiB |
| 合计 | 7168 | 896 MiB，即0.875 GiB |

缓存长度包括已经计算过 K/V 的提示词和生成位置。它不一定等于“提示词长度＋界面已经显示的输出数”：刚采样出的 token，通常还没有经过下一次前向，也就还没有它自己的 KV。

如果三条请求接下来各完成一个新位置的前向，缓存合计增加：

$$
3\times128\ \mathrm{KiB}=384\ \mathrm{KiB}
$$

因此，活跃请求数不变时，缓存也会继续增长。只检查请求刚进入服务时能否放下，还不够。

## 2. 不等长请求，应该对长度求和

把上面的过程写成公式。若所有层配置相同，K/V 维度相同，且每条序列单独保存缓存：

$$
M_{payload}=2LH_{kv}d_{head}s\sum_{i=1}^{R}T_i
$$

其中，`R` 是同时保留缓存的序列数，`T_i` 是第 `i` 条序列已缓存的位置数，`s` 是每个元素的字节数。只有长度都相同时，才可以把求和写成 `R × T`。

用最长长度乘请求数，可能高估不等长请求的有效数据；直接使用平均长度则需要明确，这个平均是**当前这些请求**的平均，不能随便换成数据集的平均输入长度。

这个公式只计算有效 KV。服务显存还包括权重、临时激活、workspace、通信缓冲、图捕获占用、分配粒度及缓存元数据。某些框架还会提前预留较大的 KV 池，所以监控里的显存占用不一定随着一条请求结束而马上下降；释放的 block 可以回到池内供下个请求使用。

另外，**总量除以 TP 度数不一定就是每张卡的 KV**。这取决于 KV head 如何分片、是否复制，以及流水线如何分层。不同层的缓存长度或结构不同时，也要逐层计算。滑动窗口、MLA 和线性注意力需要各自的状态模型。

## 3. 分页：有效数据没有消失，分配方式变了

假设一个 KV block 能存16个位置。对当前缓存长度为19和33的两条请求：

| 请求 | 有效位置 | 所需 block | 分配容量 | 尾部空位 |
| --- | ---: | ---: | ---: | ---: |
| D | 19 | 2 | 32 | 13 |
| E | 33 | 3 | 48 | 15 |
| 合计 | 52 | 5 | 80 | 28 |

计算使用向上取整：

$$
n_i=\left\lceil\frac{T_i}{b}\right\rceil,
\qquad
M_{blocks}=m_{token}\,b\sum_i n_i
$$

在本例中，有效数据为 `52 × 128 KiB = 6.5 MiB`，已分配 block 容量为 `80 × 128 KiB = 10 MiB`。多出的3.5 MiB是这些 block 中暂未使用的空间，还没有计算 block table 等元数据。

这不表示分页比连续布局必然多用3.5 MiB。比较对象应是实际分配策略：如果原先按最大长度为每条请求预留很大的连续区域，按需分配小 block 就可能节省大量预留空间。分页自身仍有尾部浪费；简单顺序填充、没有共享或额外预留时，每条序列尾部最多空 `b - 1` 个位置。

逻辑位置可以这样映射：

```text
请求 D：逻辑 block 0 → 物理 block 7
        逻辑 block 1 → 物理 block 2

请求 E：逻辑 block 0 → 物理 block 4
        逻辑 block 1 → 物理 block 9
        逻辑 block 2 → 物理 block 1
```

模型仍然按逻辑顺序访问上下文，物理 block 不要求连续。**同样52个有效位置，仅仅分页不会降低其 K/V 元素数量**。量化改变每个元素的存储成本；前缀共享减少重复存储；它们与分页解决的是不同层面的问题。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)详细介绍了这套逻辑到物理的映射。

## 4. 有空 block，不一定马上接收新请求

内存管理解决“数据放哪里”，调度解决“这一轮计算谁”。判断能否接入请求，至少要同时看：

- 能否容纳已有及即将增长的 KV
- 是否达到活跃请求数限制
- 本轮还能处理多少 token
- 调度策略如何安排 Prefill、Decode 与等待请求

这里的 token 预算是待计算位置数，不是预计生成长度。例如，8条请求各做一个标准 Decode 位置，就是8个待计算 token；再加入一段长度1000的 Prefill，合计1008个位置。这个计数可用于资源限制，**不代表 Prefill 和 Decode 的每个 token 耗时相同**。

请求数量也不等于负载大小。两条请求可能是短上下文 Decode，也可能各有数万 token 的 Prefill，计算形态和缓存需求差别很大。

## 5. 手推一次连续批处理

为看清边界，设定一个简化调度规则：最多2条活跃请求，每轮最多计算5个位置，单次 Prefill chunk 最多4个位置。

A、B已经完成 Prefill，分别还需输出2个、4个 token。C刚到达，提示词长6，需要输出2个 token。下面的输出编号只表示本例中的剩余生成进度。

| 调度轮次 | 请求 A | 请求 B | 请求 C | 本轮待计算位置 |
| --- | --- | --- | --- | ---: |
| 1 | Decode，输出 A1 | Decode，输出 B1 | 等待 | 2 |
| 2 | Decode，输出 A2，结束 | Decode，输出 B2 | 等待 | 2 |
| 3 | 已退出 | Decode，输出 B3 | Prefill 前4个位置 | 5 |
| 4 | 已退出 | Decode，输出 B4，结束 | Prefill 后2个位置，输出 C1 | 3 |
| 5 | 已退出 | 已退出 | 输入 C1，Decode 输出 C2，结束 | 1 |

C没有在第1轮插入，是因为活跃请求数已满，虽然本轮 token 预算尚有剩余。A结束后，下一轮才让C进入；不用等待B也完成，这就是迭代边界上调整 batch 的意义。

C的两个 Prefill chunk 属于同一上下文，第二块要访问第一块的 KV。完成最后一块后，其最后位置的 logits 可以产生 C1；生成 C2 时，才把 C1 作为新输入做增量前向。

这张表没有给每轮设置毫秒数。第3轮有 Prefill，不会因为轮次编号相邻，就和第1轮耗时相等。它也不是某个框架当前调度器的逐行复刻，只是用于区分请求数、token预算、缓存与迭代边界。

连续批处理允许在调度边界加入请求，**不保证任何新请求只等待一轮**，也不意味着可以在任意正在运行的 kernel 中途插入工作。资源不足或策略限制仍会造成排队。

## 6. 拆小 Prefill，为什么可能一边改善、一边变慢

长 Prefill 若一次占用设备较久，正在生成的请求可能迟迟得不到下一次 Decode。把 Prefill 拆成 chunk，可以给调度器更多安排其他请求的机会。

但 chunk 变小也可能带来更多调度和启动开销，改变矩阵乘的形态，并让该请求经历多轮才能得到首 token。所以需要同时观察新请求的首 token 延迟和已有请求的输出间隔。不是把 chunk 调得越小越好，也不能只凭一张调度表就宣称吞吐提升。

[DistServe](https://arxiv.org/abs/2401.09670)分析了 Prefill 与 Decode 共享资源时的干扰。这里借用这个问题帮助理解取舍，不复用论文的实验数字，也不把特定系统的结果当成所有硬件上的定律。

## 7. TTFT、TPOT 与 ITL：先固定计时口径

本文使用客户端时间戳：请求发出为 `t_0`，第一个输出 token 到达为 `t_1`，最后一个到达为 `t_G`，输出数为 `G`。

$$
\mathrm{TTFT}=t_1-t_0
$$

它可以包含网络、服务端排队、输入处理、Prefill和采样等，不能直接等同于设备上的 Prefill kernel 时间。服务端内部指标可能从另一个时间点开始，比较前要核对定义。

对于 `G > 1`：

$$
\mathrm{TPOT}=\frac{t_G-t_1}{G-1}
$$

分母是首 token 后的间隔数。只有一个输出 token 时，没有后续间隔，本文记为“不适用”，不把它当成0毫秒的快速 Decode。

取教学时间戳 `[300, 310, 320, 420]` 毫秒，请求在0毫秒发出：

```text
TTFT = 300 ms
相邻输出间隔 ITL = [10, 10, 100] ms
TPOT = (420 - 300) / 3 = 40 ms
```

平均40毫秒，并不能说明每个间隔都接近40毫秒；最后一次已等了100毫秒。请求级 TPOT 的分位数，与逐间隔 ITL 的分位数，也不是同一组样本。多个请求的 TPOT 若先各算平均再平均，还会与按全部间隔加权的结果不同。

如果把端到端时间也定义到**末 token 到达**，同一请求满足：

$$
T_{last}=\mathrm{TTFT}+(G-1)\mathrm{TPOT}
$$

本例为 `300 + 3 × 40 = 420` 毫秒。这是同一计时口径下的恒等式，不是无需运行就能预测延迟的模型。若工具计时到连接结束或最终统计事件，还可能多出尾部开销。一个流式响应块可能包含多个 token；没有逐 token 时间戳时，也不能把每个响应块的间隔直接当成精确 ITL。

## 8. 一个只用 Python 标准库的小练习

下面复核前面的显存与时间计算，不加载模型，不模拟设备执行。可以直接复制运行。

```python
from math import ceil, isclose

KIB, MIB, GIB = 1024, 1024**2, 1024**3
layers, kv_heads, head_dim, element_bytes = 32, 8, 128, 2
bytes_per_token = 2 * layers * kv_heads * head_dim * element_bytes
assert bytes_per_token == 128 * KIB

lengths = [4096, 2048, 1024]
payload = sum(lengths) * bytes_per_token
assert payload == 896 * MIB
assert payload / GIB == 0.875
print(f"three-request KV payload: {payload / MIB:.0f} MiB")

lengths, block_size = [19, 33], 16
blocks = sum(ceil(t / block_size) for t in lengths)
capacity = blocks * block_size
unused = capacity - sum(lengths)
assert (blocks, capacity, unused) == (5, 80, 28)
print(f"paged capacity: {capacity * bytes_per_token / MIB:.1f} MiB")
print(f"valid payload: {sum(lengths) * bytes_per_token / MIB:.1f} MiB")

request_start_ms = 0
arrivals_ms = [300, 310, 320, 420]
ttft = arrivals_ms[0] - request_start_ms
itl = [b - a for a, b in zip(arrivals_ms, arrivals_ms[1:])]
tpot = sum(itl) / len(itl) if itl else None
assert ttft == 300 and itl == [10, 10, 100] and tpot == 40
assert isclose(ttft + len(itl) * tpot, arrivals_ms[-1] - request_start_ms)
print(f"TTFT: {ttft} ms; ITL: {itl} ms; TPOT: {tpot:.0f} ms")
```

本地运行得到：

```text
three-request KV payload: 896 MiB
paged capacity: 10.0 MiB
valid payload: 6.5 MiB
TTFT: 300 ms; ITL: [10, 10, 100] ms; TPOT: 40 ms
```

可以把 block size 改成8，观察尾部空位；再把第一个到达时间固定，只增大最后一次间隔，观察 TTFT 与 TPOT。改动只说明定义和算术关系，不构成真实性能优化的证据。

## 9. 自测：先回答，再展开

1. 三条请求缓存长度不同时，为什么不能直接用“并发数×最大长度”作为有效 KV？
2. 教学模型中，KV head 从8变成32，其他条件不变，有效 KV 如何变化？
3. block size 为16，长度19的请求增长到20，需要新分配 block 吗？
4. C已经到达、每轮 token 预算也有剩余，为什么仍可能排队？
5. Prefill拆成两块后，第二块是否可以不看第一块？
6. TPOT为40毫秒，是否说明每个输出间隔都不超过40毫秒？

<details markdown="1">
<summary>展开参考答案</summary>

1. 每条请求有效缓存按自己的实际长度增长，应对长度求和。最大长度乘请求数可以作为另一种容量上界或分配策略的账，但不等于有效数据。
2. 增加到四倍。这里没有改变分片、共享和元素字节数。
3. 不需要，两个block已有32个位置的容量。增长到33时才需要第三个block；有效数据增长与新分配block不是同一个时刻。
4. 还可能受活跃请求数、KV容量、未来增长预留和调度策略限制；token预算不是唯一条件。
5. 不能。因果上下文仍要保留，拆分改变处理批次，不改变允许访问的历史。
6. 不能。本例三个间隔为10、10、100毫秒，平均仍为40毫秒。需要查看间隔分布及相应口径的尾延迟。

</details>

## 参考资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：KV block、逻辑到物理映射、迭代级调度及内存共享
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)：TTFT/TPOT需求与Prefill、Decode资源干扰

这些资料用于核对原理。本文的配置、调度表和时间戳都是教学例子；具体框架的调度边界、缓存预留与指标实现，需要结合版本继续检查。
