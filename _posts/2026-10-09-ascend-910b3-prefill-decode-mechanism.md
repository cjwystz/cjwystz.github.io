---
layout: post
title: 拆到算子看:910B3 上 prefill 和 decode 到底在争什么
date: 2026-10-09 00:00:00 +0800
categories: [inference, llm-serving]
tags: [prefill-decode-disaggregation, ascend-910b3, msprof, profiling, interference, performance-model, blog]
description: 上一篇在 910B3 上测了 PD 分离的配比效果——3P:4D 吞吐反超一体化 11%,只给 1 张 prefill 卡会把 TTFT p99 打到 9 秒。但那些都是端到端数字,回答不了"为什么"。这篇往下拆一层:用昇腾 msprof 把 Cube/Vector/MTE 三类执行单元分开观测,给出 prefill 与 decode 资源争抢的硬件级证据——同一个 MatMul 算子,prefill 负载下 Cube 占用 0.92,decode 负载下只有 0.14,而搬数通道 MTE2 被打满到 0.99。
toc:
  beginning: true
---

> **2026-10-08 实验复核补充：**原文 10.2× 的计时包含全设备等待，不能直接解释为 decode 自身延迟；利用率画像也不足以单独确定内存争抢根因。详见[后续实验复核](/blog/2026/ascend-910b3-inference-audit-negative-results/)。


> 承接上一篇《在 910B3 上把 LLM 推理的"前菜"和"主菜"分开炒——PD 分离实测》。
> 硬件:Atlas 800T A2,单卡 Ascend 910B3(64GB HBM)。软件:CANN 9.0.0 + torch_npu 2.10,profiling 用自带的 msprof。
> 结论先行:**同一个 MatMul 算子,prefill 负载下 Cube 占用 0.92、decode 负载下只有 0.14;decode 侧数据搬运通道 MTE2 被打到 0.99。** 两者在卡上同时跑,decode 每步延迟从 0.222ms 涨到 2.264ms——**慢 10.2 倍**。PD 分离论文里那句"prefill 干扰 decode",在 NPU 上可以被拆成具体的执行单元争抢,而不是一个含糊的定性断言。

---

## 一、为什么要往下拆一层

上一篇测了四种配比(1P:6D / 2P:5D / 3P:4D / 不分离),端到端指标(TTFT、TPOT、吞吐)都拿到了,结论也清楚:

- 只给 1 张 prefill 卡,rps=4 时 TTFT p99 冲到 **9018ms**(不分离基线才 111ms);
- 3P:4D 在 rps=8 下吞吐 **1684 tok/s**,比 7 卡不分离高 11%,单卡效率也高 11%。

但这些都是**结果**。它回答不了三个"为什么":

1. 为什么 prefill 卡数少会让尾延迟爆?是算力排队,还是传输堵了?
2. 为什么 decode 不被 prefill 打断就能更省资源?TPOT 全程稳在 15–18ms,这个"稳"的物理来源是什么?
3. 为什么最优配比是 3P:4D 而不是别的?能不能不靠扫参数、直接算出来?

这三个问题的共同答案是:**需要一个干扰模型**——知道 prefill 和 decode 各自吃哪种硬件资源、一起吃时怎么抢,才可能解析地推出配比,而不是靠实验枚举。

而在做模型之前,得先确认一件事:**这些资源在 910B3 上到底能不能被观测到?** 今天就把这一步做完了。

---

## 二、工具:msprof 能把执行单元拆到什么粒度

昇腾 NPU 的执行结构和 NVIDIA GPU 不一样。GPU 上你主要看一个 SM 占用率,里面算力、搬数、缓存混在一起;910B3 上 AI Core 内部是**分单元**的:

- **Cube**:矩阵乘单元(算 GEMM 的地方);
- **Vector**:向量单元(逐元素运算、softmax 这类);
- **MTE1/2/3**:三条数据搬运通道(MTE2 = 全局内存→L1/UB,MTE3 = 反向搬出);
- **Scalar**:控制流。

`msprof op`(即 msopprof)支持按这几类单元分别采集:

```bash
msprof op --application="python3 xxx.py" \
  --output=./prof \
  --aic-metrics=ArithmeticUtilization,PipeUtilization,L2Cache,ResourceConflictRatio,Memory \
  --launch-count=20 --kill=off
```

`PipeUtilization` 这一组导出的 CSV 里,每一行是一个 AI Core block,列里直接给了:

```
aic_cube_time / aic_cube_ratio          Cube 占用率
aiv_vec_time  / aiv_vec_ratio           Vector 占用率
aic_mte1_ratio / aic_mte2_ratio / aic_mte3_ratio    三条搬运通道占用率
aiv_mte2_active_bw / aiv_mte3_active_bw  带宽绝对值 (GB/s)
aic_scalar_cube_stall_time / ...         stall 按等待对象分解
```

另外 `ResourceConflictRatio` 直接给了 bank 冲突、资源冲突、各单元 wait ratio——**这些是干扰的现成量化指标**,不用自己造。

这一层粒度是关键:如果只能看到"AI Core 总占用 80%",你无法区分它是算力满了还是搬数满了;分开之后,"prefill 吃 Cube、decode 吃 MTE2"才有可能是**可验证的命题**而不是假设。

---

## 三、实验一:同一算子,两种负载,资源占用完全相反

为了把变量控干净,我没有直接上 sglang,而是用 torch_npu 写了两个受控负载,**都用 MatMul 作为核心算子**,只改 batch 形状:

**prefill-like**(模拟大 batch 一次性并行算完):

```python
X = torch.randn(4096, 4096, dtype=torch.float16, device="npu:1")
W = torch.randn(4096, 4096, dtype=torch.float16, device="npu:1")
Q = torch.randn(2048, 2048, dtype=torch.float16, device="npu:1")
K = torch.randn(2048, 2048, dtype=torch.float16, device="npu:1")
for _ in range(20):
    Y = X @ W                       # 大 GEMM
    S = Q @ K.transpose(-1, -2)     # attention score
    P = torch.softmax(S.float(), dim=-1)
```

**decode-like**(模拟 batch=1、每步回头读一大片 KV cache):

```python
H = 8192
K_cache = torch.randn(16384, H, dtype=torch.float16, device="npu:1")  # ~256MB
q = torch.randn(1, H, dtype=torch.float16, device="npu:1")
for _ in range(50):
    score = q @ K_cache.transpose(0, 1)   # 读 256MB,几乎无算力
    score = torch.softmax(score.float(), dim=-1)
```

采集结果——两边都被识别成 MatMul,但走的是不同实现(prefill 命中 `MatMulV3_ND_ND_ND_ND_FP16`,decode 命中 `MatMulV2_ND_ND_FP16`),各 block 均值如下:

| 指标 | prefill-like | decode-like | 差异 |
|---|---|---|---|
| **Cube 占用率** | **0.923** | **0.142** | **6.5×** |
| **MTE2(搬入)占用率** | 0.866 | **0.990** | decode 打满 |
| MTE1 占用率 | 0.790 | 0.439 | |

<img src="/assets/img/blog/mech_pipes.png" alt="Cube 与 MTE 占用对比" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

这张图是今天最想要的东西。它说的是:

- **prefill 是 Cube-bound**:算力单元几乎打满(0.92),搬数通道也忙(0.87)但没到极限——瓶颈在算;
- **decode 是 MTE2-bound**:Cube 只有 0.14(算力大量闲置!),但搬数通道打到 0.99——瓶颈完全在搬 KV;
- 两者**占用的是不同的资源池**。这正是 PD 分离能work的物理前提:如果把两种负载放同一张卡,一个吃 Cube、一个吃 MTE2,理论上"应该"能互补……但下面实验二会告诉你,现实没这么美好。

顺便一个细节:decode 侧 Cube 闲到 86%。**这说明在 decode 阶段,910B3 的算力是被浪费的**——这也是 AFD(attention-FFN 分离)和 decode 侧混部优化想吃的红利。

---

## 四、实验二:两者真在一张卡上跑,decode 被拖慢 10.2 倍

上面是"各自单独跑"的画像。真实 serving 里它们是同时在的,所以要测争抢。用 torch_npu 的双 stream 构造混部:

```python
s1 = torch.npu.Stream()
s2 = torch.npu.Stream()

# decode 基线:单独跑 200 步
for _ in range(N): decode_step()

# 混部:prefill 在 stream1 持续压,decode 在 stream2 计时
with torch.npu.stream(s1):
    for _ in range(N * 3): prefill_step()
with torch.npu.stream(s2):
    for _ in range(N): decode_step()
```

结果:

| 场景 | decode 每步延迟 | 倍数 |
|---|---|---|
| 单独跑 | **0.222 ms** | 1.0× |
| 与 prefill 混部 | **2.264 ms** | **10.2×** |

<img src="/assets/img/blog/mech_interf.png" alt="干扰延迟对比" style="display:block;margin:1em auto;max-width:60%;height:auto;border-radius:6px;" />

**10.2 倍**。这就是一体化(collocated)serving 里 TPOT 抖动的物理来源,也是为什么 Sarathi-Serve 要用 chunked-prefill 把大 prefill 切小、插进 decode 空隙——不切的话,decode 的每一步都可能被一个长 prefill 顶住。

对照实验一的画像看,这个 10.2× 并不意外,但**原因不是算力**:prefill 吃 Cube、decode 吃 MTE2,资源池本来错开。真正的争抢点在**共享的下游**——两者都要读写全局内存(HBM)和 L2,而 decode 恰恰是 MTE2-bound 的,它的搬数通道一旦被 prefill 的流量挤占,就直接体现为每步延迟暴涨。

这也解释了上一篇里那个现象:**1P:6D 配置下 TPOT 依然稳在 16.3ms,爆的是 TTFT p99**。因为 decode 卡独占,没人跟它抢 MTE2 和 HBM;而 prefill 只有一张卡,请求在那里排队,所以首 token 延迟炸。分离之后各自资源独占,干扰被物理消除——**这正是 PD 分离的价值所在,不是调度技巧,而是资源隔离。**

---

## 五、方法论上的一个诚实边界

必须说清楚一个限制:**`msprof op` 是 kernel-replay 模式**——它把每个 kernel 单独重放来采计数器。所以我拿到的 pipe 占用画像是"这个 kernel 孤立执行时"的资源消耗,**不包含运行时的真实争抢信息**。

证据是:我对混部脚本(`mixed.py`)同样跑了 msprof op,decode 侧 MatMul 的画像依然是 cube 0.146 / mte2 0.990——和它单独跑时几乎一样。争抢没有体现在 kernel 画像里,**争抢体现在 wall-clock 延迟里(那个 10.2×)**。

这意味着正确的组合方式是:

- **单 kernel 画像**告诉你"每个算子吃哪种资源"→ 用来建机理模型;
- **wall-clock / 服务级采样**告诉你"一起吃时慢了多少"→ 用来标定和验证模型。

服务级采样这条路(`msprof --aic-mode=sample-based` + `--sys-hardware-mem`,可对着运行中的 sglang 服务采 HBM/LLC/DDR)命令格式我已经验证可用,但还没在真实 PD 服务上跑过——这是下一步。

另外一个已知缺口:`--aic-metrics=Memory` 导出的 HBM 带宽绝对值(`aic_main_mem_read_bw` 等)在 MatMul 上返回 NA,需要配合 `ArithmeticUtilization` 或换 metric 组才能填满。带宽绝对值对量化"争抢了多少 GB/s"很关键,是建模前要先解决的一个小问题。

---

## 六、下一步:从画像到干扰模型

今天做完的是可行性验证——**三项机理(算力/搬运/缓存)在 910B3 上都可观测**,而且比 GPU 细一个量级(Cube 与 Vector 分离,三条 MTE 通道分开)。有了这个基础,干扰模型才可能做得出机理而不只是拟合曲线。

计划中的路径:

1. **受控二维扫描**:在真实 serving 里扫 (prefill batch × decode batch) 网格,同时记录 pipe 占用与 wall-clock,得到干扰曲面 I(b_p, b_d);
2. **机理归因**:把干扰分解为 共享 HBM 带宽争抢 + L2 污染 + 调度队列阻塞 三项,每项用 msprof 的指标标定;
3. **held-out 验证**:模型在没扫过的配置(换模型规模、换序列长度)上预测,验证是否可外推——这是"模型"与"拟合"的分界线;
4. **导出决策**:有了状态依赖的服务率,就能解析推导 colocated vs PD 的相变边界,以及最优配比,替代上一篇那种实验枚举。

第 4 步才是研究价值的落点:现在所有人都靠扫参数选配比(包括上一篇的我自己),如果能从干扰模型**解析地**算出配比,那就不只是"在国产卡上复现了一个结论",而是给出了一个别人在 GPU 上也用得上的方法。

而 910B3 的 Cube/Vector/MTE 分离观测,恰好让机理归因这一步比在 GPU 上做更有可能——GPU 上你只有一个 SM 占用率,分不清争抢发生在算力还是搬数。

---

## 附:今天的四个实验清单

| # | 实验 | 方法 | 结果 |
|---|---|---|---|
| 1 | 工具链摸底 | `msprof --help` / `msprof op --help` | Cube/Vector/MTE/L2/冲突率/HBM 均可采 |
| 2 | 最小 kernel 验证 | 4096² matmul + softmax,`--launch-count=8` | CSV 真实导出,数值可读 |
| 3 | 单负载资源画像 | prefill-like vs decode-like | Cube 0.92 vs 0.14,MTE2 0.87 vs 0.99 |
| 4 | 干扰实测 | torch_npu 双 stream 混部 | decode 0.222ms → 2.264ms,**10.2×** |

全部实验在单卡 910B3 上完成(卡 1,卡 0 留给别人的任务),脚本与原始 profiling 数据在 node200 `~/chenjiawei/pd-exp/`(`mini_kernel.py` / `prefill_like.py` / `decode_like.py` / `mixed.py`,profile 结果在 `prof_test/` `prof_prefill/` `prof_decode/` `prof_mixed/`)。
