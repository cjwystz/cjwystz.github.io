---
layout: post
title: RAG 优化文生图——不动模型，修复提示词错位
date: 2025-11-15 09:00:00 +0800
categories: [research]
tags: [multimodal, rag, t2i]
thumbnail: assets/img/blog/rag-t2i.svg
---

*论文笔记：《RAG-Optimized Text-to-Image Generation for Consumer Platforms》，发表于 WISE 2025 Workshop（清华 BNRist）。代码：[github.com/Dadada66666/T2I](https://github.com/Dadada66666/T2I)。*

## 问题

在文生图（T2I）系统中充当"提示词工程师"的小参数 LLM，经常在**隐性知识**上翻车——那些对人来说显而易见（主体数量、空间布局、风格预期）、但用户短短一句提示词里没有的细节。只靠 LLM 自己改写提示词补不上这个缺口，因为模型自身的知识量太小。

## 我们的做法

不改 T2I 模型、不做任何重训练，而是在 T2I 流水线之前插入一个 **RAG 前置的提示词优化阶段**：

1. **检索**：两段式混合检索流水线——Tavily（网络搜索）加上 ChromaDB 里的 BGE-m3 向量检索，针对目标场景拉取相关的多模态知识。
2. **提示词构建**：把检索到的上下文组装进改写提示词，让小 LLM "知道"它原本缺失的隐性知识。
3. **部署**：基于 ollama + ComfyUI 构建——只多一次推理调用，T2I 模型零改动。

## 结果

平均生成质量得分相比纯 LLM 改写提升 **+114%**。该框架是模型无关的：任意 T2I 后端、任意小 LLM 都能直接接入这条流水线。

## 几点体会

- 提示词工程本质上是个信息问题：把缺失的知识喂给模型，它自己的能力就够用了。
- RAG 可以作为生成任务的**预处理阶段**使用，不只是问答场景的专利。
- 不动基座模型让系统足够便宜，可以跑在消费级硬件上。
