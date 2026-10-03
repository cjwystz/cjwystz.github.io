---
layout: post
title: 把一次真实事故变成 AI 考题：国产 infra benchmark 的第一道题
date: 2026-10-03 09:00:00 +0800
categories: [engineering]
tags: [benchmark, harbor, ai-infra, python, debugging]
description: 把在国产 GPU 平台适配中遇到的一个真实 vendor 库升级事故，改写成一道 Harbor 格式的 agentic benchmark 题——包括 bug 的 Python import 机制原理、题目环境设计，以及"不要标准答案、只要判分器"的 verify 哲学。
toc:
  beginning: true
---

TL;DR：

- 一位在头部大模型公司做 RL 的朋友找我出题：**把真实踩过的 infra 坑做成 agentic benchmark**——给 AI agent 的"工程能力考题"，格式用 Harbor；
- 第一道题选了我在国产 GPU 平台适配中的真实事故：vendor 库升级（torch2.8 → torch2.10 构建）后 `import transformer_engine.pytorch` 静默炸掉，根因是一个关于 `sys.modules` 别名的条件恢复 bug；
- 本文讲三件事：事故本身的技术原理、怎么把它变成可自动判分的题、以及为什么"不要 GT、只要 verify"。

## 1. 缘起：RL 时代需要什么样的"题"

先交代背景。做 agent/RL 训练的人现在最缺的不是模型，是**任务数据**——而且是"能自动判分"的任务。这是范式转变：

- **老范式（给 GT）**：训练数据 = 题目 + 标准答案，模型模仿答案。问题是复杂工程问题的标注成本极高，人类自己都未必判得对错；
- **新范式（给 verifier）**：训练数据 = 题目 + 判分器。不告诉模型正确答案，只告诉它"怎么算对"，让它自己探索，判分器给奖励。

对出题人的要求也随之改变：**不需要你会解这道题，只需要你会判分**。而"解不出来但有明确判据"的题，恰恰是系统方向最富集的矿——部署踩坑、升级翻车、性能回退，每个都有现成的复现环境和客观的判据（服务通不通、测试过不过、速度快不快）。

朋友的原话："把你们工作里觉得困难的问题转成题，难度无所谓，只要题干无歧义、答案可以 verify。"于是有了这第一道题。

## 2. 事故还原：一次 vendor 库升级引发的静默故障

事故发生在国产 GPU 平台的训练框架适配工作中。架构是两层：

```
上层：TEFL（TransformerEngine 封装层）
  └─ transformer_engine.pytorch.*   ← 训练脚本 import 的公开 API
        └─ import transformer_engine_torch   ← "dispatch 模块"，负责转发
下层：vendor 预编译 C 扩展（.so，厂商闭源交付，按 torch 版本出多个 build）
  └─ transformer_engine_torch.so   ← 注意：和 dispatch 模块同名！
```

某天 CI 把 vendor 包从 torch2.8 构建切到 torch2.10 构建，冒烟测试开始挂：

```
ModuleNotFoundError: No module named 'transformer_engine_torch'
```

诡异的是：torch2.8 构建一切正常，两个构建的 .so 都好好躺在磁盘上，而且**回退就能好**——但不能回退，torch2.10 构建修了另一个我们需要的数值 bug。

## 3. Bug 原理：三层 Python import 机制叠加的坑

这道题的难度全在"诊断"——修复本身只有两行，但定位要理解三件事。

### 3.1 sys.modules 是 import 系统的注册表

`import transformer_engine_torch` 时，Python 先查 `sys.modules` 字典：命中就直接返回缓存的模块对象，不命中才走查找-加载-注册流程。TEFL 的 dispatch 模块是包初始化时**手动注册**进去的：

```python
# transformer_engine/plugin/core/_module_setup.py
dispatch = types.ModuleType("transformer_engine_torch")
dispatch.__getattr__ = forward_to_vendor_backend   # 属性访问转发给 vendor 后端
sys.modules["transformer_engine_torch"] = dispatch  # 注册别名
```

所以业务代码 `import transformer_engine_torch` 拿到的其实是这个转发器。

### 3.2 单阶段初始化的 C 扩展会"自我注册"

加载 vendor 的 `.so` 用的是按路径加载：

```python
spec = importlib.util.spec_from_file_location("transformer_engine_torch", so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
```

关键点：**老式单阶段初始化（single-phase init）的 C 扩展，在 exec_module 期间会把自己的模块对象塞进 `sys.modules`**——名字取它的 PyInit 符号名 `transformer_engine_torch`。也就是说，vendor 的 torch2.8 构建加载时会**挤掉**我们刚注册的 dispatch 别名。

而 torch2.10 构建（多阶段初始化）**不会**自我注册。同一厂商、不同构建，行为不同。

### 3.3 条件恢复：只对"会自我注册"的世界做了防御

当年的 loader 作者知道扩展会挤别名，于是写了"先存后还"：

```python
saved_owner = sys.modules.pop(_ALIAS, None)   # 先把 dispatch 别名摘下来保管
spec.loader.exec_module(mod)                  # 加载扩展（可能把别名挤走）
...
if sys.modules.get(_ALIAS) is mod:            # 只有"被挤了"才恢复
    if saved_owner is not None:
        sys.modules[_ALIAS] = saved_owner
```

torch2.8 时代：扩展自我注册 → 条件命中 → 别名恢复 → 一切正常。

torch2.10 时代：扩展**不**自我注册 → 条件不命中 → **`pop` 出去的别名永远没还回来** → 下游 `import transformer_engine_torch` 找不到模块 → ModuleNotFoundError。

防御逻辑和攻击面错位了：代码防的是"扩展会挤别名"，真实事故却是"别名被自己 pop 了之后没人还"。修复就是让恢复**无条件**发生：

```python
if saved_owner is not None:
    sys.modules[_ALIAS] = saved_owner
else:
    sys.modules.pop(_ALIAS, None)
```

## 4. 从事故到考题：Harbor 任务设计

真实事故的环境（内部 vendor 包）带不出来，所以题目对现场做了**等比缩放重建**——保留事故的完整因果链，去掉所有外部依赖。

### 4.1 环境：条件必须给全

```
environment/
├── Dockerfile                    # 构建期用 gcc 现编两个"vendor 扩展"
├── vendor_ext/
│   ├── tex_torch28.c             # 单阶段初始化（会自我注册）
│   ├── tex_torch210.c            # 多阶段初始化（不自我注册）
│   └── build_all.sh              # 产出两个 build 目录 + installed 符号链接
└── app/                          # 完整可跑的 TEFL 封装层
    ├── run_smoke.py              # 冒烟入口（必挂）
    ├── logs/crash_torch210.log   # 完整报错堆栈
    └── transformer_engine/...    # dispatch 安装、vendor loader、公开 API
```

两个"vendor 扩展"是 Docker 构建时现场编译的真 C 扩展，行为差异（是否自我注册）由 C 代码保证——这样判分脚本可以把"扩展行为"当作**环境事实**来钉，而不信任任何 Python 层的状态。容器里自带复现所需的全部东西，agent 不需要访问任何外部资源。

### 4.2 判分：行为契约 + 反作弊

`tests/test.sh` → pytest，**每个检查都在全新子进程里跑**（import 状态不能跨检查泄漏，和真实训练任务启动条件一致）。判分点设计：

| 检查 | 性质 |
|---|---|
| torch2.10 扩展不自我注册、torch2.8 自我注册 | **反作弊**：钉死环境事实，agent 重编/替换 .so 会被抓 |
| `run_smoke.py` 输出 SMOKE OK | 原始症状修复 |
| dispatch 别名 = 包初始化时安装的那个模块对象（`is` 同一性） | 防止自己 shadow 一个假模块顶名 |
| vendor 模块的 `__file__` 真实路径指向活跃 build 的 .so | 防止偷换 build |
| 通过别名访问能转发到 vendor 后端（git id 对得上） | 转发链完整性 |
| 重复加载返回同一模块对象（扩展不被执行两次） | 幂等性 |
| 以上全部对 torch2.8 build 同样成立 | 防回归：不许"修了新版砸旧版" |

输出是多维的 `rewards.json`（每个检查独立计分）——比单一 0/1 更适合 RL 训练，agent 做到哪一步拿哪步的分。

### 4.3 Oracle 解

`solution/solve.sh` 就是第 3.3 节那个两行 patch + 双 build 冒烟自检。它的存在不是给训练用，是**证明这道题可解**——这是任务入库前的质量门禁。

## 5. 什么算一道好题（目前的体会）

1. **难度来自诊断，不来自工作量**。修复两行，定位要懂 import 机制。这种题区分的是"真懂系统"和"会补全代码"；
2. **判分锚点要钉在环境事实上**，不钉在 agent 产出上。本题的锚是两个 C 扩展的自我注册行为——agent 改不了它，一改就被发现；
3. **反作弊和正判据一样重要**。同名模块的题目天然诱导"shadow 一个假模块"的作弊路径，所以判分里有对象同一性、真实路径、幂等性三道锁；
4. **多维奖励 > 单维通过**。对部分完成的 agent 给部分分，RL 信号更密；
5. **可解性必须自证**。oracle 跑通是入库门槛，不是可选项——不可解的题对训练是纯噪声。

本地四步验证（复现 bug → 判分拦截 → oracle 修复 → 判分全绿）已通过，接下来用真实 agent 端到端跑。后续候选题也在排队：多网卡 GLOO 组网超时（docker-compose 多容器）、vendor 通信库"伪 OOM"日志诊断、dispatcher 静默回退导致训练吞吐腰斩（这题判分就是跑分——"速度即 verify"）。

## 6. 和主线工作的关系

这件事和我平时的活儿是同一条线：在国产芯片上踩的每一个坑，以前是"解决完就翻篇"，现在多了一条出路——**沉淀成可复用的评测资产**。部署 910B 的坑、调 MoE 的坑、这次的 vendor 库升级事故，都是同一个视角：把昂贵的、非公开可得的工程经验，转成可自动判分的题目。

相关：[Harbor 框架](https://github.com/laude-institute/harbor) · 同系列：[昇腾 910B 部署 SGLang 推理服务实战](/blog/2026/sglang-ascend-910b-deploy/) · [追一个静默精度 bug：我给 SGLang 提 PR 的完整记录](/blog/2026/sglang-npu-moe-fp32-pr/)
