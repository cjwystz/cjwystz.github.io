---
layout: post
title: "给 PEFT 修一个 LoRA-FA 优化器接口：closure 为什么不能 backward？"
date: 2026-10-14 10:00:00 +0800
categories: [engineering, llm-training]
tags: [peft, lora, lorafa, pytorch, autograd, open-source]
description: 从 CPU 最小复现到梯度上下文修复，记录 PEFT issue #3958 与 PR #3966 的测试证据、维护者批准和重叠 PR 检查。
toc:
  beginning: true
---

这次向 Hugging Face PEFT 提交了 [issue #3958](https://github.com/huggingface/peft/issues/3958) 和 [PR #3966](https://github.com/huggingface/peft/pull/3966)。问题很具体：LoRA-FA 优化器声明支持 `step(closure)`，但执行 closure 时关闭了梯度记录，导致标准训练 closure 中的反向传播失败。

**更新于 2026 年 10 月 16 日：PR #3966 已于北京时间 16:26 合并，维护者审查批准，线上 CI 全部通过。** 本文记录一个优化器 API 正确性修复，不声称改善训练吞吐或模型效果。

## 基础知识：LoRA、LoRA-FA 与优化器

LoRA 用低秩矩阵表示权重更新：`ΔW = s · B · A`。底层模型权重通常冻结，训练主要更新适配器参数。默认初始化时 B 为零，因此刚加入适配器时，权重增量为零。

LoRA-FA 固定 A，仅优化 B，并利用 A 构造梯度投影。这里不修改投影公式、学习率、权重衰减或适配器结构，关注的是优化器如何执行训练回调。有关方法本身，可以参考 [LoRA-FA 论文](https://arxiv.org/abs/2308.03303)与 [PEFT 优化器文档](https://huggingface.co/docs/peft/en/developer_guides/lora#lora-fa-optimizer)。

普通训练循环先由用户计算梯度，再更新参数：

```python
optimizer.zero_grad()
loss = model(inputs).loss
loss.backward()
optimizer.step()
```

另一种接口把计算 loss 和梯度的操作放进 closure，也就是一个交给优化器调用的函数：

```python
def closure():
    optimizer.zero_grad()
    loss = model(inputs).loss
    loss.backward()
    return loss

optimizer.step(closure)
```

[PyTorch 官方示例](https://docs.pytorch.org/docs/2.14/optim.html#optimizer-step-closure)就是这种结构。并非每种优化器都需要重复执行 closure；本次只要求 LoRA-FA 已公开提供的可选接口能正确执行一次。

## 为什么会失败

检查的 PEFT main 版本是 `eac46efeef6e5f63ecc38de5c9148513df602b29`。[源码](https://github.com/huggingface/peft/blob/eac46efeef6e5f63ecc38de5c9148513df602b29/src/peft/optimizers/lorafa.py#L84-L94)中，`step()` 被 `@torch.no_grad()` 修饰，同时直接调用 closure：

```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        loss = closure()
    # 后续参数更新
```

参数更新通常不需要记录计算图，因此外层 `no_grad` 是合理的。但 closure 要执行模型前向和反向传播，必须记录计算图。这个上下文覆盖整个函数，导致 closure 中的 loss 不带梯度信息，`backward()` 无法继续。

这与冻结基础模型权重不同：即使 A 和基础权重冻结，训练中的 B 仍需要通过前向计算获得梯度。

## CPU 最小复现

无需模型下载或集群，用一个 Linear 层和 rank-2 LoRA 即可：

```python
import torch
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer

torch.manual_seed(42)
model = get_peft_model(
    torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)),
    LoraConfig(target_modules=["0"], r=2, lora_alpha=2),
).train()
optimizer = create_lorafa_optimizer(model, r=2, lora_alpha=2, lr=1e-3)
x = torch.ones(2, 4)


def closure():
    optimizer.zero_grad()
    loss = model(x).square().mean()
    print(torch.is_grad_enabled(), loss.requires_grad)
    loss.backward()
    return loss


optimizer.step(closure)
```

未修复代码输出 `False False`，随后报错：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

复现环境为 macOS arm64、Python 3.12.14、PyTorch 2.14.1、Transformers 5.19.0、Accelerate 1.15.0，PEFT 从上述源码安装，版本为 `0.21.3.dev0`。

对照实验中，同一模型使用普通 LoRA-FA 前向/反向后调用 `step()`，或使用 AdamW 的 `step(closure)`，均能更新权重。在 closure 内显式开启梯度也能恢复正常。因此证据指向梯度上下文，而不是模型、数据或设备配置。

## PR 做了什么

生产代码仅把 closure 调用包进梯度上下文：

```python
loss = None
if closure is not None:
    with torch.enable_grad():
        loss = closure()
```

保留外层 `@torch.no_grad()`：进入 closure 时开启梯度，离开后恢复，后续参数更新仍不记录计算图。这与 PyTorch AdamW 的处理方式一致。

最终版本修改两个文件：`src/peft/optimizers/lorafa.py` 和 `tests/test_lorafa.py`。生产代码仅增加 `torch.enable_grad()` 上下文，测试经过两轮维护者审查后收敛为一个最小回归用例。完整改动见 [已合并 PR](https://github.com/huggingface/peft/pull/3966/files)。

## 测试如何证明修复有效

最初使用普通 LoRA / rsLoRA、外层 `no_grad` 等组合，并比较多步参数更新。维护者指出，对这个梯度上下文错误，这些检查过于复杂。最终删除参考训练路径、多步比较、rsLoRA 和外层 `no_grad` 参数化，只保留普通 LoRA 的一次 closure 调用：

```python
def test_lorafa_step_with_closure():
    torch.manual_seed(42)
    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)),
        LoraConfig(target_modules=["0"], r=2, lora_alpha=4),
    )
    optimizer = create_lorafa_optimizer(model, r=2, lora_alpha=4, lr=0.001)
    x = torch.ones(2, 4)

    def closure():
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    grad = model.get_base_model()[0].lora_B["default"].weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad) > 0
```

这里没有在 `backward()` 之前断言梯度模式，而是让未修复代码真正触发用户遇到的 RuntimeError；修复后再检查 B 的梯度非空且非零。测试针对具体故障，避免把无关行为绑定到回归测试中。

| 检查 | 最终结果 |
| --- | --- |
| 移除修复，运行单个回归测试 | 在 `loss.backward()` 触发原始 RuntimeError |
| 修复后本地完整 `tests/test_lorafa.py` | 11 通过 |
| 本地 `make quality` | 通过 |
| 线上 Python 3.11–3.14 × Ubuntu / Windows | 8 项测试任务全部通过 |
| 线上代码质量检查与文档构建 | 全部通过 |

新增回归测试使用 CPU；本地已有测试自动选择 Apple MPS。没有执行 CUDA/NPU、多机或完整大模型训练测试，因此不把这次 API 正确性修复描述为全套硬件兼容性或性能验证。[线上测试记录](https://github.com/huggingface/peft/actions/runs/38537785692)。

维护者 BenjaminBossan 于北京时间 10 月 16 日 14:14 [批准最终版本](https://github.com/huggingface/peft/pull/3966#pullrequestreview-5543749559)，表示 “Thanks for simplifying the test. LGTM.”；PR 随后于 16:26 合并。

## 开源流程与重叠 PR

首先提 issue，附复现并说明愿意提供修复。维护者 BenjaminBossan 于北京时间 10 月 14 日 08:50 [明确批准](https://github.com/huggingface/peft/issues/3958#issuecomment-6158378200)，随后提交 PR #3966。[PEFT 贡献规范](https://huggingface.co/docs/peft/main/en/developer_guides/contributing)要求包括草稿在内的 PR 都先取得 issue 批准。

**问题是我先发现、复现并通过 issue #3958 报告的。另一位贡献者随后引用这个 issue 提交修复。问题发现与报告、修复 PR 的提交顺序，是两件不同的事。**

这里有一个需要如实记录的检查遗漏：另一位贡献者 YashwanthNomula 已于 10 月 13 日 22:19 提交 [PR #3959](https://github.com/huggingface/peft/pull/3959)，引用同一个 issue，采用相同修复思路。它在维护者批准前被机器人以“缺少已批准 issue”为由关闭，技术方案没有被人工否定。检查时只关注开放 PR，漏掉了这个关闭后仍可能重开的实现。

最初发布时，#3959 关闭，#3966 开放；10 月 16 日更新时，#3966 已合并。我的贡献包含先前的问题发现、可运行复现与 issue 报告，以及获批准后提交的修复和回归测试；另一位作者较早提交了修复 PR，不表示他较早发现或报告了问题。同时，不能因为机器人关闭就认为另一位作者放弃了。最终上游合并了 #3966。这提醒我：重复检查要包括关闭的 PR、issue 时间线和作者的接手声明。

## AI 协助与贡献状态

OpenAI Codex / GPT 参与源码调查、复现、实现、测试执行和文稿整理，issue 与 PR 已披露 AI 协助。测试结果来自本地实际执行，不能用自动生成的文字替代验证。提交者仍需理解改动、检查每一行并能够回应审查；PEFT 明确要求人对 AI 辅助贡献负责。

目前可准确表述为：**Hugging Face PEFT 开源贡献者：发现并报告 LoRA-FA closure 梯度上下文错误，完成修复与回归测试，经维护者审查、线上 CI 验证后合并至上游。**
