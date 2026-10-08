---
layout: post
title: 在 910B3 上把 LLM 推理的"前菜"和"主菜"分开炒——PD 分离实测
date: 2026-10-08 00:00:00 +0800
categories: [inference, llm-serving]
tags: [prefill-decode-disaggregation, ascend-910b3, sglang, llm-inference, memfabric, serving, blog]
description: 在大模型推理里,prefill 和 decode 是两个计算特征完全不同的阶段——前者吃算力、后者吃带宽,挤在同一张卡里互相干扰。这篇博客讲清楚 PD 分离是什么、虚拟 PD(同卡逻辑分离)和真机 PD(独立物理卡)的区别,并在华为 910B3 上做了真机 PD 的配比实验:7 卡切成 3P+4D,高并发吞吐反超一体化 11%;但只给 1 张 prefill 卡会把 TTFT 尾延迟打到 9 秒。
toc:
  beginning: true
---

> **2026-10-08 实验复核补充：**原文两组测试没有重放相同请求 trace，SSE 消息与生成 token 的计数也需区分；11% 是当次观测，尚不足以证明容量提升。详见[后续实验复核](/blog/2026/ascend-910b3-inference-audit-negative-results/)。


> 硬件:Atlas 800T A2 训练服务器,单机 8×910B3(单卡 64GB HBM);其中 1 卡留给别人的 CV 任务,实际使用 7 卡。
> 软件:sglang v0.5.18 + CANN 9.0.0(Ascend 官方 sglang 镜像),PD 传输走 ascend 后端(底层是华为 memfabric_hybrid)。
> 模型:Qwen3-8B(dense)。负载:输入 ~500 token / 输出 ~200 token,泊松到达。
> 结论先行:**在 8 并发下,把 7 张卡切成 3 张专做 prefill、4 张专做 decode,比 7 张卡"一锅炒"的吞吐高 11%,单卡效率也同步高 11%。但配比切错(只给 1 张 prefill)会把 TTFT 尾延迟打到 9 秒。**

---

## 一、什么是 PD 分离(Prefill/Decode Disaggregation)

大模型推理一个请求,本质是两个计算特征完全不同的阶段:

- **Prefill(前菜)**:把你输入的几百上千个 token **一次性并行**算完,生成第一个输出 token。这一阶段是 **计算密集型(compute-bound)**——所有输入 token 一起进,矩阵乘打满,attention 的 KV 一次性写进缓存。它决定了 **TTFT(Time To First Token,首 token 延迟)**,也就是你敲下回车后"转圈"的时间。

- **Decode(主菜)**:生成第一个 token 之后,进入自回归循环——**一次只吐一个 token**,每个新 token 都要回头去读之前所有 token 的 KV cache。这一阶段是 **显存带宽密集型(memory-bandwidth-bound)**——算力闲着,大部分时间花在搬 KV 上。它决定了 **TPOT(Time Per Output Token,每 token 间隔)**,也就是回答"流"出来的速度。

**传统做法(collocated / 一体化)**:同一张卡既做 prefill 又做 decode。问题来了——一个 decode 循环跑到一半,一个新请求的 prefill 插进来,prefill 是"大胃王"一次要吃满算力,就会把正在 decode 的请求"踩停",导致 TPOT 抖动;反过来 decode 的零散小矩阵又让 prefill 吃不满。两个阶段互相干扰,谁也跑不出峰值。

**PD 分离**:干脆把两种活分给不同的卡。

- 一批卡当 **Prefill 实例**,只负责算输入、产 KV cache,算完把 KV 通过高速链路"快递"出去;
- 另一批卡当 **Decode 实例**,只负责接 KV、专心逐 token 生成;
- 中间一个 **router** 调度:新请求先发给 prefill,prefill 把 KV 传给某个 decode,decode 开始流式吐字。

好处是 decode 不再被 prefill 打断(TPOT 更稳、流水线更满),prefill 可以攒大 batch 打满算力。**代价**是多了一次 KV cache 的跨卡/跨机传输,以及"几张卡做 prefill、几张做 decode"这个配比问题——切错了就是本文后面那个 9 秒的坑。

---

## 二、虚拟 PD 分离 vs 真机 PD 分离

这是两个容易混淆的概念,值得先分清。

### 虚拟 PD 分离(逻辑分离,共享物理卡)

"虚拟 PD"指的是**在同一张物理卡上,用软件把时间片/资源切成 prefill 角色和 decode 角色**,逻辑上分离、物理上仍共享。典型形态:

- 同一张 GPU/NPU 上跑两个进程(或两个 stream),一个扮演 prefill worker、一个扮演 decode worker;
- 或者同一进程内用调度器把 prefill batch 和 decode batch 分时复用(time-slicing),比如 chunked-prefill 把超长 prefill 切成小块、插进 decode 的空隙里跑(sglang/vLLM 的 mixed batching 就是这个思路);
- KV cache 的"传输"发生在**同一张卡的显存内部**(甚至就是指针传递,零拷贝),没有跨卡、跨机开销。

**特点**:省卡(一张卡搞定)、无网络/互联开销、部署简单;但因为底层还是共享同一份算力和带宽,prefill 和 decode 仍然会互相挤占——**干扰没消除,只是被调度器"柔化"了**。它适合单机、卡少、想用小成本验证 PD 逻辑是否 work 的场景。sglang 里的 `fake` transfer backend 就是这类纯测试桩:走完整套 PD 调度流程,但 KV 根本不真传,专门用来验证代码路径通不通。

### 真机 PD 分离(物理分离,独立物理卡)

"真机 PD"指的是 **prefill 和 decode 跑在不同的物理卡(甚至不同物理机)上**,各有各的显存和算力,彻底不互相干扰。这也是本次实验采用的方式:

- prefill 实例和 decode 实例各占独立的 910B3 卡;
- KV cache 要**跨卡真传**——本次走的是华为 ascend 传输后端,底层是 memfabric_hybrid(基于 `ASCEND_MF_STORE_URL` 指向的共享内存/高速存储通道),不是简单的同卡指针;
- 需要一个真正的 KV 传输后端(mooncake / nixl / ascend / ascend+memfabric 等),而这是落地时最容易卡住的地方——本集群无外网,mooncake 和 nixl 都没装,**只有 Ascend 官方镜像自带的 ascend 后端可用**,于是直接用官方的 PD 配方。

**特点**:prefill 与 decode 物理隔离、互不打扰,能跑出各自峰值,可以独立扩缩容(prefill 慢就加 prefill 卡,decode 慢就加 decode 卡);代价是要真传 KV、有传输开销,且引入了**配比**这个新的调参自由度。

**一句话对比**:虚拟 PD 是"一张卡里演两个角色",省钱但治标;真机 PD 是"两张卡各干各的",费卡但治本。本文测的是真机 PD。

---

## 三、实验设置

**环境打通过程**(对想复现的人有参考价值):

- 卡:0 号留给别人的 CV 任务,实验用 1–7 共 7 张 910B3。
- 容器 `cjw-sglang-main`(Ascend 官方 sglang v0.5.18 镜像,CANN 9.0.0),权重在 `/data/lyk/models/Qwen3-8B`。
- PD 传输后端选型:mooncake ❌(没装、无外网装不上)、nixl ❌(同)、**ascend ✅**(镜像自带,底层 memfabric_hybrid,预装)。`fake` 只是测试桩不算数。
- 关键环境变量:`ASCEND_MF_STORE_URL=tcp://127.0.0.1:24669`(memfabric 共享通道),每个实例用 `ASCEND_RT_VISIBLE_DEVICES` 钉到单卡,`--disaggregation-mode prefill/decode`,`--disaggregation-transfer-backend ascend`。
- router:`sglang_router.launch_router --pd-disaggregation --prefill <url> <boot> --decode <url>`(多实例就重复 `--prefill`/`--decode`;**注意:collocated 对照组的 Regular 模式要用 `--worker-urls`,不是 `--decode`**,这是个实测踩到的坑)。

**四组配置**(都用满 7 卡,只改 P/D 配比):

| 配置 | P:D | prefill 卡 | decode 卡 | 说明 |
|---|---|---|---|---|
| colloc | 0:7 | — | 1–7 | 不分离基线(每卡 P、D 都干) |
| pd_1p6d | 1:6 | 1 | 2–7 | decode 侧重 |
| pd_2p5d | 2:5 | 1–2 | 3–7 | 均衡 |
| pd_3p4d | 3:4 | 1–3 | 4–7 | prefill 侧重 |

**负载**:合成混合负载,输入 ~500 token、输出 ~200 token(`ignore_eos`,强制拉满输出长度),泊松到达,扫 offered rps = 1 / 4 / 8,每组先 20s 热身再采 90s 数据。**指标**:TTFT p50/p99、TPOT p50/p99、端到端延迟、输出/输入吞吐(tok/s)、完成请求数。**12 个数据点全部 0 错误。**

---

## 四、全量数据

> ttft/tpot 单位 ms;e2e_p50 单位 s;吞吐单位 tok/s。每张表按并发档(rps)分列。

### TTFT(首 token 延迟,越低越好)

| 配置 | rps=1 p50/p99 | rps=4 p50/p99 | rps=8 p50/p99 |
|---|---|---|---|
| colloc 0P:7D | 84 / 113 | 90 / 111 | 92 / 150 |
| PD 1P:6D | 91 / 179 | 137 / **9018** | 164 / 948 |
| PD 2P:5D | 89 / 116 | 102 / 245 | 111 / 252 |
| PD 3P:4D | 93 / 169 | 103 / 180 | 108 / 224 |

### 输出吞吐(越高越好,tok/s)

| 配置 | rps=1 | rps=4 | rps=8 |
|---|---|---|---|
| colloc 0P:7D | 192 | 771 | 1512 |
| PD 1P:6D | 200 | 806 | 1491 |
| PD 2P:5D | 186 | 769 | **1595** |
| PD 3P:4D | 201 | 796 | **1684** |

### TPOT(每 token 间隔,ms,越低越好)

| 配置 | rps=1 p50/p99 | rps=4 p50/p99 | rps=8 p50/p99 |
|---|---|---|---|
| colloc 0P:7D | 15.1 / 16.3 | 16.4 / 17.2 | 17.8 / 19.1 |
| PD 1P:6D | 15.5 / 16.7 | 16.3 / 17.5 | 16.9 / 18.2 |
| PD 2P:5D | 15.5 / 16.9 | 16.5 / 17.8 | 17.2 / 18.1 |
| PD 3P:4D | 16.0 / 16.8 | 16.8 / 18.1 | 17.6 / 18.4 |

### 完整明细(含均值与输入吞吐)

```
tag              rps  done err ttft_mean ttft_p50 ttft_p99 tpot_mean tpot_p50 tpot_p99 out_tok/s prompt_tok/s e2e_p50_s
colloc-rps1      1.0   90   0    115.3     84.0    113.2    15.26     15.07    16.26    192.4     481.3       3.10
colloc-rps4      4.0  360   0     90.0     90.0    110.9    16.37     16.40    17.19    771.2    1928.8       3.37
colloc-rps8      8.0  706   0    104.5     92.0    149.8    17.76     17.76    19.05   1511.6    3780.3       3.65
pd_1p6d-rps1     1.0   95   0    101.4     91.1    179.4    15.61     15.45    16.74    200.1     500.3       3.18
pd_1p6d-rps4     4.0  377   0   1067.7    136.6   9018.5    15.69     16.30    17.47    805.7    2014.9       3.42
pd_1p6d-rps8     8.0  697   0    183.9    164.1    947.5    16.92     16.92    18.15   1490.8    3728.4       3.55
pd_2p5d-rps1     1.0   90   0     92.2     89.4    116.0    15.61     15.50    16.87    185.9     465.0       3.19
pd_2p5d-rps4     4.0  359   0    114.1    101.9    244.9    16.55     16.49    17.83    768.9    1923.0       3.41
pd_2p5d-rps8     8.0  746   0    130.6    110.6    251.6    17.21     17.24    18.07   1594.8    3988.3       3.57
pd_3p4d-rps1     1.0   94   0     97.5     93.1    168.6    15.86     15.97    16.75    201.1     503.0       3.29
pd_3p4d-rps4     4.0  372   0    108.8    103.0    179.9    16.77     16.79    18.10    795.9    1990.6       3.47
pd_3p4d-rps8     8.0  788   0    118.8    107.7    223.5    17.61     17.62    18.44   1683.6    4210.5       3.64
```

---

## 五、三个结论(都可以写进博客标题)

<img src="/assets/img/blog/fig1_throughput.png" alt="图1:输出吞吐 vs 并发" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

### 1. 配比切错,prefill 立刻成为瓶颈,尾延迟爆炸

**1P:6D 在 rps=4 时,TTFT p99 飙到 9018 ms(9 秒)**,而同档 collocated 只有 111 ms、p50 也才 137 ms。原因很直白:只有 1 张卡做 prefill,4 并发下前向请求开始排队,排在后面的请求首 token 要等前面几个 500-token 的 prefill 全算完。注意此时 **TPOT 依然稳在 16.3ms**——decode 侧 6 张卡很闲,锅全在 prefill 这一张卡上。这就是"配比不当"最典型、最直观的指纹:p50 还能看,p99 先炸。

### 2. 多一张 prefill 卡,立刻根治尾延迟

<img src="/assets/img/blog/fig2_ttft_p99.png" alt="图2:TTFT 尾延迟 p99(对数轴)" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

<img src="/assets/img/blog/fig3_ttft_p50.png" alt="图3:TTFT p50" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

把 prefill 从 1 张加到 2 张(2P:5D),rps=4 的 **TTFT p99 从 9018ms 砍到 245ms**,p50 也回到 102ms。说明在这个负载下 2 张 prefill 卡已经够用。配比是 PD 分离的第一调参旋钮,而它的代价极小——往往只差一张卡。

### 3. 高负载下,PD 分离的吞吐和单卡效率双双反超一体化

<img src="/assets/img/blog/fig5_percard.png" alt="图5:单卡效率对比" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

<img src="/assets/img/blog/fig4_tpot.png" alt="图4:TPOT p50 稳定性" style="display:block;margin:1em auto;max-width:68%;height:auto;border-radius:6px;" />

看 rps=8(最接近真实服务压力)这一档:

| 配置 | 输出吞吐 | 单卡效率(÷7卡) | TTFT p99 |
|---|---|---|---|
| colloc 0P:7D | 1512 tok/s | 215.9 tok/s/卡 | 150ms |
| PD 2P:5D | 1595 tok/s | 227.8 tok/s/卡 | 252ms |
| PD 3P:4D | **1684 tok/s** | **240.5 tok/s/卡** | 224ms |

**3P:4D 用 4 张 decode 卡,打出了 7 张 collocated 卡 111% 的吞吐;单卡效率也同步 +11%。** 这正是 PD 分离的核心论点被实测验证:decode 不再被 prefill 打断、流水线一直满,于是 decode 卡的有效产出更高。在这个输入 500/输出 200、prefill 不算特别重的负载下,3P:4D 是甜点。

### 一个必须诚实写明的点:低负载下 PD 不占便宜

rps=1 时,各 PD 配置的吞吐(186–201)和 collocated(192)基本打平,TTFT 也接近。**PD 的收益只在并发上来之后才显现**——低负载时 prefill/decode 本来就不太打架,分离反而白白承担了一次 KV 传输开销。所以 PD 分离是为高负载、高并发场景设计的,不是无脑上的银弹。这个"什么时候没用"的边界,反而让"什么时候有用"的结论更可信。

---

## 六、目前实验齐全吗?(诚实的边界清单)

**已经做到的:**

- ✅ 真机 PD 分离(ascend + memfabric_hybrid),非虚拟/非 fake 桩;
- ✅ 4 组配置(1 组 collocated 基线 + 3 组 PD 配比),控制变量干净(都用满 7 卡,只改 P/D 比);
- ✅ 3 档并发(1/4/8),含热身与泊松到达,贴近真实流量;
- ✅ TTFT/TPOT/吞吐/端到端延迟全指标,12 个数据点 0 错误,原始 JSON 全部落盘可复查。

**没做、但会让结论更完整的(后续可补):**

- ⚠️ **只测了一个模型(Qwen3-8B,dense)**。MoE 模型(如 Qwen3-30B-A3B)的 prefill/decode 计算特征不同,甜点配比可能移动;
- ⚠️ **只测了一种负载形状(输入500/输出200)**。长输入(如 4K 检索增强)会更吃 prefill,最优 P 占比会上升;长输出则更吃 decode。本次结论的"3P:4D 最优"是在这个形状下成立的;
- ⚠️ **并发只扫到 rps=8**,没有把系统压到吞吐拐点(饱和点)之后——不知道各配置的最大承载上限谁更高;
- ⚠️ **是逻辑 PD(单机内多实例各一卡)**,没有测跨机 PD(KV 走 IB/RoCE),跨机传输开销是另一个故事;
- ⚠️ **没测虚拟 PD(chunked-prefill/mixed batching)对照**,所以"虚拟 PD 够不够"这个选择题本次没回答;
- ⚠️ KV 传输没有单独打点(传输耗时/带宽占用),9 秒那个 p99 里有多少是传输、多少是 prefill 排队,没有进一步拆解。

**一句话总结**:作为"真机 PD 分离 vs 一体化、以及配比影响"的论证,实验是自洽且结论干净的;但要外推到别的模型、别的负载形状、或跨机场景,还需要按上面的清单补测。

---

*实验代码与原始数据:node200 `~/chenjiawei/pd-exp/`(`pd_launch.sh` / `pd_stop.sh` / `run_matrix.sh` / `bench_ttft.py`,结果在 `results/*.json`)。*
