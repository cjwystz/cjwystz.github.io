---
layout: post
title: 解剖一道顶级 agentic 考题：FrontierSWE 推理优化题的设计艺术
date: 2026-10-04 00:00:00 +0800
categories: [engineering]
tags: [benchmark, harbor, ai-infra, sglang, 论文调研]
description: 逐文件拆解 FrontierSWE 的 inference-system-optimization——B200 上 SGLang 服务优化，让 agent 去打一个已经拉满配置的 baseline。从任务结构、判分流水线、防作弊武器库三个层面分析它为什么是好题，以及对国产 infra benchmark 的七点可借鉴设计。
toc:
  beginning: true
---

TL;DR：

- 逐文件读完了 Proximal Labs 的 FrontierSWE 里 `inference-system-optimization` 这道题（B200 + SGLang + Qwen3.5-4B，让 AI agent 优化推理服务速度）；
- 它和一般的"跑个分"benchmark 完全是两个物种：**baseline 本身已经是配置天花板，agent 的唯一出路是写 kernel、改引擎源码、动模型**；
- 判分不是"结果对不对"，而是一条四阶段流水线（baseline 测速 → candidate 正确性门禁 → candidate 测速 → baseline 复测），每一环都针对 agent 的作弊路径做了防御；
- 本文按"结构 → 判分 → 防作弊 → 对国产 benchmark 的借鉴"四层解剖。

## 1. 题目形态：为什么是"打 baseline"而不是"达到指标"

先看任务设定，一句话：**给你一个 B200 上跑 SGLang 服务 Qwen3.5-4B 的环境，和一个已经调优到配置极限的 baseline，你要比它快**。

baseline 的配置是：`--kv-cache-dtype fp8_e4m3`（FP8 KV cache）+ `--speculative-algorithm NEXTN`（MTP 投机解码）+ mamba scheduler V2 + CUDA graphs + `mem-fraction-static 0.88` 精细显存分配。instruction 里直接明牌：

> To beat it, you need to go beyond configuration — custom kernels, SGLang source modifications, or model surgery.

这是这道题最核心的设计决策。对比一下常见的失败设计：

| 设计 | 结果 |
|---|---|
| "把吞吐优化到 X tokens/s" | agent 背一遍 vLLM/SGLang 调优手册就能过，考的是知识检索 |
| "baseline 是个烂配置，去优化它" | 等于把答案写进题里，agent 只要认出"哦这是没开 flash attention" |
| **"baseline 已经是天花板，去打它"** | 配置级知识全部失效，只能靠真功夫：profiling 找瓶颈、写自定义 kernel、改框架源码 |

这直接决定了题目的难度曲线：拿 cookbook 里的标准操作（换量化、调 batch、开 CUDA graph）得分 ≈ 1.0x（和 baseline 一样快，零 headroom），想拿高分必须进入"无标准答案"的领域。

## 2. 结构解剖：四个角色各司其职

```
inference-system-optimization/
├── task.toml              # 声明：frontier 难度、4h agent、B200、断网
├── instruction.md         # 题面：给 agent 看的全部信息
├── environment/
│   ├── Dockerfile         # 完整环境：CUDA 12.8 + torch 2.10 + SGLang + 全套 kernel 工具
│   └── workspace/         # agent 的工作区（dev 工具链）
│       ├── submission/launch_server.sh   # ← agent 唯一要交付的东西
│       ├── run_dev_bench.py              # 公开 dev benchmark（和隐藏判分不同负载）
│       ├── verify_serving.py             # 快速自检
│       └── timer.sh                      # 计时守护进程
├── solution/              # Oracle 解（fp8 + torch.compile + tuned CUDA graphs）
└── tests/                 # 判分（agent 看不到）
    ├── test.sh            # 预检 + 反作弊扫描 + 环境恢复
    ├── launch_baseline.sh # verifier 自有的 baseline 启动脚本
    ├── compute_reward.py  # 1049 行：四阶段判分流水线
    └── prompts.jsonl      # 1166 条隐藏 prompt（正确性门禁的弹药）
```

几个值得单独说的设计：

**交付物被刻意收敛成一个文件**。agent 最终只需要留下 `/app/submission/launch_server.sh`——verifier 用它启动你的服务。这比"交出你改过的所有代码"高明：它把评测面收敛到一个可执行的入口，防止 agent 依赖环境里的脏状态（instruction 里还专门警告：直接改 site-packages 的修改 replay 时会丢，必须在 launch_server.sh 里重新应用）。

**给 agent 配了完整的 dev 工具链，且明确告诉它 dev 负载和隐藏判分负载不同**。`run_dev_bench.py` 是公开的，可以用来迭代；但 instruction 明说"The hidden verifier uses different workloads"——既给了反馈回路（不然 agent 瞎跑），又防止过拟合到 dev 负载。

**计时守护进程**（`timer.sh`）：agent 可以随时 `cat /app/.timer/remaining_secs` 查剩余时间，30/10/5 分钟有警告文件。这不是便利设施，是评测设计的一部分——**它把"时间管理"变成了被考察的能力**：agent 必须自己决定什么时候停止优化、留下足够时间跑最终验证。

## 3. 判分流水线：四阶段，每阶段都有防御目标

`compute_reward.py` 的主流程是四阶段，我按顺序拆：

### Phase 1：Baseline 测速 + 收集参考输出

先起 baseline 服务（verifier 自己的 `launch_baseline.sh`），在隐藏负载上测速，同时在 1166 条隐藏 prompt 上收集 greedy 输出作为正确性参考。

**防御目标**：baseline 是 verifier 自带的，agent 碰不到它——防止 agent 通过污染 baseline 来"提高自己的相对速度"。

### Phase 2：Candidate 正确性门禁 → 测速

起 agent 的服务，先在同样的 1166 条 prompt 上收集输出，做 **token 级比对**（按空白分词，计算最长公共前缀比例），要求平均匹配率 ≥ 0.95。**不过门禁，直接 0 分，速度根本不测。**

这是整个判分里最重要的一道闸。它防的是所有"牺牲正确性换速度"的作弊路径：乱搞量化把模型搞废、投机解码太激进导致输出漂移、改 kernel 引入数值错误。注意阈值设的 0.95 而不是 1.0——给合法的数值抖动（fp8 vs bf16 的合理差异）留了空间，这是跑过真实数据后校准出来的工程判断。

过了门禁才测速：隐藏负载矩阵是 ISL/OSL 四象限（短入短出/长入短出/短入长出/长入长出）+ 一个推理题 + 并发负载（4 并发、8 并发）。**单点快没用，必须全象限都快**——专门防"只对某种负载过拟合"（比如只对长文本优化了 prefill，短文本反而慢了）。

### Phase 3：Baseline 复测

candidate 测完，把 baseline 再测一遍。如果两次 baseline 结果漂移超过 30%，打 variance flag。

**防御目标**：检测环境不稳定（GPU 热降频、agent 残留的进程污染了环境）。这是真实跑过 benchmark 的人才会加的检查——他们注释里写了，之前踩过"残留 triton cache 导致 5-8% 方差"的坑，所以现在 test.sh 里会**先恢复干净的 SGLang 包（tar 快照还原）、清所有 `__pycache__` 和 triton cache、pkill 残留的 GPU 进程**，再开始判分。

### Phase 4：几何平均加速比

所有负载的加速比（baseline 中位延迟 / candidate 中位延迟）做**几何平均**。几何平均而不是算术平均的选择是有讲究的：它惩罚短板——一个负载 3x、另一个 0.3x，算术平均 1.65 看着还行，几何平均只有 0.95 直接打回原形。**鼓励均衡优化，惩罚顾此失彼。**

## 4. 防作弊武器库：七个机制

把散在各处的防御集中看，这道题对 agent 作弊路径的封锁是体系化的：

| 机制 | 位置 | 防什么 |
|---|---|---|
| 隐藏 prompt 集（1166 条，含对抗/退化输入） | tests/ | 过拟合 dev 负载、背题 |
| token 级正确性门禁（≥0.95） | compute_reward.py | 牺牲质量换速度 |
| baseline 由 verifier 私有启动 | launch_baseline.sh | 污染 baseline 抬高相对分 |
| 源码扫描（grep `/tests/`、`compute_reward`、`reward.json`） | test.sh 预检 | 直接读判分逻辑/答案 |
| 环境恢复（tar 还原 + 清缓存 + 杀残留进程） | test.sh | 环境污染、脏状态影响判分 |
| baseline 复测 + 漂移检测（>30% 打 flag） | Phase 3 | 环境不稳定、计时操纵 |
| 提交物收敛为单入口脚本 | instruction.md | 依赖不可复现的环境脏状态 |

还有一个有意思的细节：`test.sh` 会往 SGLang 的启动命令里**自动注入 `--skip-server-warmup`**——因为 SGLang 内部 warmup 有硬编码的 600s 超时，重配置下会误杀自己的服务，而 verifier 自己实现了更稳的三段式就绪检查（TCP 端口 → `/v1/models` → 真实生成一次）。这说明这套判分是被反复真实运行打磨过的，不是纸面设计。

## 5. 对国产 infra benchmark 的七点借鉴

对着这道题，我们能直接抄走的：

**1. "打满配 baseline"是难度设计的第一性原理。** 我们之前那道 vendor-alias 题、以及所有"诊断类"题，baseline 其实就是"坏了的系统"，agent 修好就满分——天花板有限。性能题的正确形态是 baseline 已经很好，逼 agent 进入无标准答案区。国产卡版本：给一个官方推荐配置全开的 vLLM-Ascend / SGLang NPU baseline，让 agent 去打它。

**2. 正确性门禁先于速度，且阈值要校准不要拍脑袋。** 0.95 是跑出来的。我们出国产卡题时，token match 阈值必须在真卡上实测（不同硬件数值行为不同，昇腾和 NVIDIA 的 fp 行为差异会直接影响这个值）。

**3. 判分负载矩阵要覆盖 ISL/OSL 四象限 + 并发。** 这是 InferenceMAX 的标准方法论，直接搬。国产卡推理题尤其需要——不同后端在 prefill-heavy 和 decode-heavy 下的瓶颈完全不同（CSA/FlashAttention 行为差异大）。

**4. 环境恢复是判分可信度的一半。** tar 快照还原 + 清 pycache/triton cache + pkill GPU 残留进程，这三件套我们一道题都不能少。我们自己踩过"公共池互踩"的坑（松江 PPU 128 卡 benchmark 被别人 pkill 误伤），对这点应该有切肤之痛。

**5. 提交物收敛为单入口。** 我们的 gpt350-mfu-race 用了 `entrypoint.sh`，是对的做法；但要补上"site-packages 修改 replay 会丢"的明确警告和 launch 时重新应用的机制。

**6. 计时守护进程让"时间管理"成为考点。** 成本几乎为零（一个 20 行 shell 循环），但把"什么时候该停止优化、什么时候该留时间验证"这个真实工程判断纳入了评测。

**7. 几何平均 + 复测漂移检测 = 分数的抗操纵性。** 特别是复测——我们 PPU benchmark 运维里被"公共池互踩"搞过，环境漂移是国产集群的常态，这条对我们比对他们还重要。

## 6. 一个冷静的评估：哪些不能直接抄

- **B200 级别的环境依赖**：这题镜像 80GB、128GB 内存、1 张 B200。国产卡版本要重新设计资源规格（910B 的显存 64GB，模型选型要缩）；
- **job.yaml 里那套多 agent 混跑**（claude-code / codex / gemini / qwen 同题并发）：这是他们运营 leaderboard 的基础设施，我们阶段还不需要；
- **1166 条 prompt 的生成管线**（`generate_prompts.py` 794 行）：可以用，但正确性门禁的 prompt 里要加中文和国产场景（他们的 prompt 偏英文通用）。

## 7. 和我们自己的题对照

我们已交付的 `metax-gemm-silent-fallback` 是**诊断题**（修复一个静默 bug，0/1 判分），这道是**优化题**（连续分数，打满配 baseline）——正好是两个互补的题型。FrontierSWE 的 17 道题里也是这个配比：实现类（写类型检查器、写编译器）+ 性能类（FFmpeg 优化、推理优化）+ ML 研究类。我们的 benchmark 也应该按这个矩阵铺开，而不是全是诊断题。

参考：[FrontierSWE 官网](https://frontierswe.com) · [本题源码](https://github.com/cjwystz/frontier-swe-chenjiawei/tree/main/tasks/inference-system-optimization) · 我们的实践：[把一次真实事故变成 AI 考题](/blog/2026/harbor-infra-benchmark-first-task/)
