---
layout: post
title: "零拷贝也要交手续费：我的 vLLM Issue 到 PR"
date: 2026-11-11 16:00:00 +0800
categories: [engineering, llm-serving]
tags: [vllm, flashinfer, kv-cache, benchmarking, open-source]
description: 从确认 FlashInfer 子任务、复现重复创建 KV 视图，到 H20 验证、性能对照和 vLLM PR #66972 的提交记录。
toc:
  beginning: true
---

这次在 vLLM 做的事说起来像套娃：**给 KV cache 的视图再加一个缓存。** 为了让代码少做几次 `transpose` 和 `split`，回归测试先多了 107 行。

11 月 11 日下午，我发出了 [PR #66972](https://github.com/vllm-project/vllm/pull/66972)。GitHub 机器人很热情，欢迎信到了，红叉也到了。翻完日志才发现，卷子还没发，红叉来自考场门口。

## 接活前，先确认别人没在写同一行

故事从 [issue #64572](https://github.com/vllm-project/vllm/issues/64572) 开始。发起人 selimfedakar 发现，FlashAttention 上已经讨论过的 K/V 视图重复构造，在其他后端也有类似情况。他把几个后端列出来，大家可以各做一块。[FlashAttention 的参考改动](https://github.com/vllm-project/vllm/pull/64571)也给了一个好起点。

我在 11 月 8 日先看相关 PR 的实际改动。有个 [PR #65885](https://github.com/vllm-project/vllm/pull/65885) 看名字和关键词很像邻居，点进文件列表才发现，没改到我要看的 FlashInfer 后端文件。**排重这件事，标题负责把人领进门，diff 才负责认人。**

当天我[在 issue 留言](https://github.com/vllm-project/vllm/issues/64572#issuecomment-6716755411)，问 FlashInfer 的 TRTLLM 子任务有没有人接，以及需要什么验证。11 月 11 日凌晨，发起人[回复确认分工](https://github.com/vllm-project/vllm/issues/64572#issuecomment-6776394351)。最后提交的范围很小：只处理 `FlashInferImpl.do_kv_cache_update` 的写入视图，其他后端各走各的。

## 显存没搬家，CPU 还在填表

K/V 视图共享 KV cache 的底层存储。原来的写入路径每次都重新推导视图，再把它们交给写入 kernel。数据没被复制，但 Python 还要创建视图对象、处理形状和步幅。这就是标题里的“手续费”。

前期我先在 CPU 上截住最后的 CUDA 调用，保留实际的 PyTorch 视图操作。对同一个缓存张量连续调用，底层存储和布局都一样，拿到的视图对象却是新建的。这一步能确认重复工作确实发生，GPU 上的数值和性能留给后面的实验。

改法是只记住当前一组 `(KV tensor, K/V views)`。同一个张量对象再来就复用，换对象就重建。这里用对象身份做判断：两个张量可以共用一块存储，视图信息却不一定相同。最后还要在 `clear_layer_kv_caches` 清掉这一组引用。退租时要把钥匙还了；不然前面的清理看着挺忙，缓存还在替显存续租。

写入 kernel 的参数、读取路径和单独的 NVFP4 FA2 写入分支都沿用原逻辑。[实际改动](https://github.com/vllm-project/vllm/pull/66972/files)分在后端、清理函数和测试三个文件里。

## 从 CPU 小实验，到 H20 上真的写一遍

我在一张 NVIDIA H20 上跑了真正调用 CUDA 写入 kernel 的回归。新测试检查反复调用是否复用视图、换张量是否重建、清理后引用能否释放，也逐项核对写入结果。原版的 FP16、BF16 和 FP8 三个用例都在“应该复用视图”这条断言上失败；改完后三个通过。这三组里，原版写入数值也与参考一致，失败的是复用检查。

接着用真实权重的 Qwen2.5-0.5B 跑前后对照：BF16 KV 的 eager、FP8 KV 的 eager，以及 BF16 KV 的 CUDA Graph。中间有一轮卡在测试脚本的 RPC 序列化，修的是脚本，改动本身没因此重写。重跑完成后，三组各 384 个生成 token，共 **1152 个 token 完全一致**，记录下来的 top logprobs 也一致。

这张 H20 的 decode 实际走 XQA。Blackwell 专用 TRTLLM-gen 和需要 SM100 的非 FA2 NVFP4 写入路径仍没有本地实测，后者的回归用例按硬件条件跳过。[验证记录和复现脚本都在 PR 里](https://github.com/vllm-project/vllm/pull/66972#validation)。

## 省了几微秒，也多付了一点手续费

性能用的是实际 CUDA writer 的**主机端提交耗时**。把同步放在计时区间外，原版和候选固定在同一个逻辑 CPU 上，按 A、B、B、A 交错运行。五种用例各做七轮，140 个计时窗口都保留。下面摘两个场景：

| 场景（BF16，4 个 KV heads） | 原版中位耗时 | 修改后中位耗时 | 配对中位变化 |
| --------------------------- | -----------: | -------------: | -----------: |
| 反复使用同一个 KV 张量      |    11.965 µs |       7.280 µs |      −39.22% |
| 两个 KV 张量轮流使用        |    11.952 µs |      12.234 µs |       +2.29% |

同一张量重复使用的四种工况，耗时下降 **37.7%–39.2%**；两个张量轮流来，每次都错过这一项缓存，反而略慢。这个坏消息也写进了 [PR](https://github.com/vllm-project/vllm/pull/66972)。表里的百分比来自逐轮配对，衡量的是这一段主机调用，不能给它升职成 GPU kernel 加速或模型吞吐提升。

## PR 发出去了，pre-commit 还在门外

补齐证据、复查有没有重叠的改动之后，我把分支推到自己的 fork。北京时间 **14:18**，[PR #66972](https://github.com/vllm-project/vllm/pull/66972) 提交到 vLLM；随后[回到 issue](https://github.com/vllm-project/vllm/issues/64572#issuecomment-6788675105)，把链接、验证范围和未覆盖的路径交代清楚，请发起人帮忙看。开发、复核和验证准备有 Codex 协助，PR 里也注明了。

三个改动文件适用的本地 pre-commit 检查通过了，远端却亮了红灯。[日志](https://github.com/vllm-project/vllm/actions/runs/42971569467/job/130361014480)里失败的是运行资格预检：它先看 PR 有没有放行标签，或者作者在 vLLM 已合并的 PR 是否达到四个。我两样都没满足，远端 pre-commit 被跳过。这张红叉记录的是门禁，没给代码打分。

截至 11 月 11 日写这篇时，PR 已提交，仍在等人工 review，尚未合并。今天省下来的几微秒有记录，没测过的路径也有记录。缓存少填了几张表，我多写了一些测试，算是把手续费交在了更划算的地方。
