---
layout: page
title: 基础知识巩固
permalink: /blog/llm-foundations/
nav: false
description: 一起从大模型的结构、计算和缓存开始，逐步理解推理系统。每篇都有公式、具体算例与巩固题。
---

这个栏目是我的大模型基础学习笔记。希望把“知道名词”推进到“能解释计算、读懂形状、判断系统瓶颈”：从 Transformer 结构开始，再逐步走到 Attention、KV Cache、服务调度与分布式推理。

每篇采用相同的学习方式：**直觉解释 → 公式与张量形状 → 具体例子 → 系统联系 → 自测与答案**。教学假设、实测结果和实现差异会分别说明。

## 已发布

{% assign foundation_posts = site.categories.foundations | sort: 'date' %}
{% for post in foundation_posts %}

- **{{ post.date | date: "%Y-%m-%d" }}**：[{{ post.title }}]({{ post.url | relative_url }})

{% endfor %}

## 后续学习路线

以下是计划主题，尚未发布。

1. Attention 的形状、mask、数值稳定与手算。
2. KV Cache、GQA、滑动窗口和显存管理。
3. Prefill／Decode 的性能模型与延迟指标。
4. Continuous batching、chunked prefill 和前缀复用。
5. 张量并行、通信与 P/D 分离。

[返回全部 blog](/blog/)
