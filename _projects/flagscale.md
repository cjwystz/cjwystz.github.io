---
layout: page
title: FlagOS / FlagScale
description: MetaX adaptation, memory optimization, and multimodal training
img:
importance: 1
category: open-source
---

**FlagScale / Megatron-LM-FL / TransformerEngine-FL**: MetaX adaptation, training framework optimization, and bug fixes

**TransformerEngine-FL**: TE 2.13+ package layout and loading-path adaptation ([PR #128](https://github.com/flagos-ai/TransformerEngine-FL/pull/128))<br>
**FlagScale**: MetaX argument-check and profiler fixes ([PR #1296](https://github.com/flagos-ai/FlagScale/pull/1296))<br>
**Megatron-LM-FL**: MetaX JIT/Dynamo memory-leak fix ([PR #185](https://github.com/flagos-ai/Megatron-LM-FL/pull/185))<br>
**Chunk CE**: Chunked vocabulary-parallel cross-entropy to reduce temporary training memory ([PR #126](https://github.com/flagos-ai/Megatron-LM-FL/pull/126))<br>
**TP overlap**: Communication–computation overlap adaptation and fixes on MetaX ([PR #139](https://github.com/flagos-ai/TransformerEngine-FL/pull/139))<br>
**Multimodal packing**: BFD / Next Fit packing algorithms; Qwen3.5 Next Fit with multimodal alignment and per-sample position encoding ([PR #1308](https://github.com/flagos-ai/FlagScale/pull/1308))<br>
