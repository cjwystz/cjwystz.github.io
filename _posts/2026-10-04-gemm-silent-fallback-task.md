---
layout: post
title: 训练慢了 13 倍，CI 却全绿：第二道 infra 考题的诞生
date: 2026-10-04 00:00:00 +0800
categories: [engineering]
tags: [benchmark, harbor, ai-infra, debugging, gemm]
description: 把"升级后 GEMM 静默回退慢路径"的真实事故做成第二道 Harbor 考题——症状是没报错、数值全对、只是慢 13 倍。讲清楚三层 CPU 模拟栈怎么让一道硬件题变成全球可跑，以及判分八项检查背后的防作弊设计。
toc:
  beginning: true
---

TL;DR：

- 真实事故：国产 GPU 平台上一次软件栈升级后，27B 模型训练吞吐掉了 13 倍，但**所有数值 CI 全绿**——GEMM 被静默路由到了慢速 Triton 回退路径，没有任何报错；
- 把它做成了 Harbor 格式的第二道题 `metax-gemm-silent-fallback`：纯 CPU、无 GPU 依赖的 Docker 镜像，docker 端到端验证通过；
- 这篇讲三件事：事故本身的排查路径、怎么把硬件事故翻译成 CPU 可复现的题、以及判分的反作弊设计。

## 1. 事故：最危险的那种故障——不报错，只是慢

事故形态是所有 infra 工程师的噩梦：

- 升级 vendor 软件栈（torch 2.8 → 2.10 对应的新扩展包）；
- 第二天训练吞吐从 ~176 TF/s 掉到 ~104 TF/s；
- 但 loss 曲线正常、数值 CI 全绿、没有任何 error log。

排查过程和教科书一样标准却耗时：profile 一遍训练，发现 **vendor 高性能 GEMM kernel 一次都没被调用**，全部 GEMM 都走在一个 JIT 回退实现上。顺着 dispatcher 往上摸，根因落在扩展加载器的一个小改动上：新版加载器要求扩展在 `exec_module` 期间**自我注册**到 `sys.modules`，否则抛 `ImportError`——而 vendor 的快路径扩展是 PEP 489 多阶段初始化，**不会**自我注册。于是加载失败，dispatcher 按设计"优雅地"回退到慢速路径，一切看起来正常，只是慢。

修复是一行：加载器在 `exec_module` 后无条件把模块装进 `sys.modules`。但定位花掉的时间以天计。

## 2. 题目化：三层 CPU 模拟栈

这道题的设计目标是把上面的事故**完整因果链**保留，同时把硬件依赖剥到零。答案是一个三层结构：

```
train.py / model.py / optim.py   ← 真训练循环（前向反向、权重更新、loss 下降）
        ↓
te.plugin.dispatcher + loader    ← 真调度链（候选链 + 静默回退 + 那个 bug）
        ↓
三个真编译的 C 扩展 .so           ← gcc 在镜像构建时现编，行为精确复刻
        ↓
stub torch（纯 Python）           ← 假 torch，2D 张量 = list of float
```

几个关键设计决策：

- **训练是真的**：模型缩到 2 层 × hidden 256、batch 64，前向 2 次 GEMM、反向 3 次 GEMM，全部经过 dispatcher；计时用真实 `perf_counter`，loss 是真矩阵乘法的结果。判分里的"两次运行 loss 逐位一致"（确定性检查）因此是真检查，不是摆设；
- **三个 .so 是"演员"**：`cublas_backend.so`（优化 C GEMM，~41ms/iter，PEP 489 不自我注册）、`triton_backend.so`（故意写慢的 JIT 风实现，~651ms/iter）、`torch_backend.so`（老式单阶段初始化，会自我注册）。13 倍的速度差由 C 代码实现差异保证，任何 x86 机器上都能复现；
- **bug 只埋在 loader 一处**：症状（慢 13 倍、数值全对）→ 观察（trace 里全是 triton）→ 根因（自我注册条件）→ 修复（一行），难度全在诊断链路，和真实事故同构。

## 3. 判分：8 项检查，0/1

第一版判分教我的是"verify 比 GT 重要"，这一版把防作弊做成了主体。8 项检查分四类：

| 类别 | 检查 | 防的是什么 |
|---|---|---|
| 正判据 | 默认配置跑通、≤500ms/iter、backend=cublas | 修复必须真的生效 |
| 契约保持 | `TE_PLUGIN_OPS` 显式指定 torch/triton 时必须被尊重 | 禁止硬编码 cublas、禁止砸掉回退链 |
| 数值/工作量审计 | 两次运行 loss 逐位一致；cublas 数值与参考实现对拍 1e-5；trace 里恰好 50 次 dispatched GEMM | 禁止跳过训练、禁止改数学换速度、禁止绕 dispatcher |
| 环境完整性 | 三个 .so 的 sha256 与构建时快照一致；训练驱动文件无直接后端 import | 禁止重编/替换 vendor 二进制、禁止作弊直通 |

特别说两个得意的设计：

- **trace 审计（第 6 项）**：profiler 记录每次 dispatched GEMM 调用，判分要求恰好 `iters × (2L + 2L-1)` 次——agent 想"少算一点换速度"会被精确抓住；
- **快照不可再生**：生成 sha256 快照的脚本在 build 完成后从镜像里删除——agent 就算篡改了 .so，也没法给自己重新盖章。

## 4. 交付的教训：docker 没封装等于没交付

这道题第一版交付时被朋友（头部大模型公司 RL 团队）打回：**"docker 没封装"**——意思是只有源码和 Dockerfile，从没真正 build 成镜像端到端跑通过。宿主机验证 ≠ 容器验证，两个典型坑当场现形：

1. `test.sh` 在运行时 `pip install pytest`——评测容器断网就直接死。修法：pytest 在 build 阶段烘进镜像；
2. 构建环境差异：宿主机有 gcc、有镜像源、有缓存，干净容器里什么都没有，apt 源慢、构建器版本老（不支持 heredoc）都得处理。

最终的交付形态也顺手固定下来了：**镜像（docker save 的 tar.gz，免 build）+ 源码包（instruction/solution/tests，判分时挂载）**，两文件缺一不可。以后每道题都按这个规格交付。

## 5. 下一步

这道题已经交付，等真 agent 的难度校准结果（如果前沿模型 1 小时内满分，就触发加难预案：减题面信号、加误导日志、埋第二个 bug）。第三道题在设计上更大——20 小时预算的 GPU 性能题：给你一个**已经调优到极限的**训练 baseline（bf16 + flash attention + fused optimizer + compile），要求在规定 token 预算内跑得更快，唯一剩下的空间是自定义 kernel——对标 FrontierSWE 的 inference 题形态，判分 = loss 门禁 + 多窗口几何平均加速比。

系列前作：[把一次真实事故变成 AI 考题：国产 infra benchmark 的第一道题](/blog/2026/harbor-infra-benchmark-first-task/) · [昇腾 910B 部署 SGLang 推理服务实战](/blog/2026/sglang-ascend-910b-deploy/) · [SGLang PR 记录](/blog/2026/sglang-npu-moe-fp32-pr/)
