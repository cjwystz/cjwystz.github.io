---
layout: post
title: RAG-Optimized Text-to-Image — Fixing Prompt Mismatch Without Touching the Model
date: 2025-11-15 09:00:00 +0800
categories: [research]
tags: [multimodal, rag, t2i]
related_posts: false
---

*Paper note: "RAG-Optimized Text-to-Image Generation for Consumer Platforms", published at the WISE 2025 Workshop (Tsinghua BNRist). Code: [github.com/Dadada66666/T2I](https://github.com/Dadada66666/T2I).*

## The problem

Small-parameter LLMs used as prompt engineers for text-to-image (T2I) systems often fail on **implicit knowledge** — details that are obvious to a human (subject counts, spatial layout, style expectations) but absent from the user's short prompt. Rewriting prompts with the LLM alone keeps the gap because the model's own knowledge is too small to fill it.

## Our approach

Instead of modifying the T2I model or retraining anything, we insert a **RAG-fronted prompt-optimization stage** before the T2I pipeline:

1. **Retrieval**: a two-stage hybrid pipeline combining Tavily (web search) with BGE-m3 embeddings in ChromaDB to pull relevant multimodal knowledge for the target scene.
2. **Prompt construction**: retrieved context is composed into the rewriting prompt, so the small LLM now "knows" the implicit knowledge it was missing.
3. **Deployment**: built on ollama + ComfyUI — one inference pass, no T2I model changes.

## Results

Average generation scores improve by **+114%** over LLM-only rewriting. The framework is model-agnostic: any T2I backend and any small LLM can drop into the pipeline.

## Takeaways

- Prompt engineering is an information problem: give the model the missing knowledge, and its own capacity suffices.
- RAG works as a **pre-processing stage** for generation tasks, not only for question answering.
- Keeping the base models untouched makes the system cheap and portable to consumer-grade hardware.
