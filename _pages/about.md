---
layout: about
title: about
permalink: /
subtitle: AI Infra Researcher &amp; Engineer · B.Eng. Information Security @ USTB

profile:
  align: right
  image: prof_pic.jpg
  image_circular: false # crops the image to make it circular
  more_info: >
    <p>2451427796@qq.com</p>
    <p>+86 186 5933 6708</p>
    <p>WeChat: _cjwystz</p>

selected_papers: true # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<style>
  /* Deep-space blue accent + subtle gradient & dot-grid texture.
     Adapts to light/dark mode via al-folio CSS variables. */
  :root {
    --global-theme-color: #3b5bdb;
  }
  body {
    background-color: var(--global-bg-color, #ffffff);
    background-image:
      linear-gradient(180deg,
        color-mix(in srgb, var(--global-bg-color, #ffffff) 88%, #3b5bdb) 0%,
        var(--global-bg-color, #ffffff) 45%),
      radial-gradient(ellipse at 85% 0%,
        color-mix(in srgb, var(--global-bg-color, #ffffff) 93%, #7048e8) 0%,
        transparent 45%);
    background-attachment: fixed;
  }
  body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(120, 130, 160, 0.08) 1px, transparent 1px);
    background-size: 22px 22px;
  }
  .navbar, .post, footer {
    position: relative;
    z-index: 1;
  }
  .profile img {
    width: 100%;
    aspect-ratio: 3 / 5;
    object-fit: cover;
    object-position: center 20%;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(59, 91, 219, 0.25);
  }
</style>

I am an **AI Infra researcher & engineer** focused on **large-model distributed training and inference systems**, memory optimization, and load scheduling. I am an undergraduate (2023–2027) in Information Security at the **University of Science & Technology Beijing**, with AI-Infra internship experience at **Qingcheng Jizhi**, contributing to the FlagOS open-source ecosystem.

My hands-on experience spans **1000-GPU-scale training and inference across heterogeneous accelerators** — Huawei Ascend 910C, MetaX C550, and Ali Zhenwu 810E — alongside NVIDIA A800/A100 clusters. I have also worked with an **eight-node NVIDIA H20 cluster**. I am a **co-first author** of an ongoing study on sequence packing and scheduling for large-model training. My work on formal-proof-guided geometric reasoning (TMLR 2027, under review) improves GRPO-trained reasoning by **+9.7%** on geometry and **+8.0%** on general math.

I am an open-source contributor to **FlagOS** and **Hugging Face PEFT**, focusing on large-model training and inference systems.

I am actively seeking a **PhD (2028–2029 Fall) in large-model systems** — distributed training/inference, memory & scheduling optimization, and embodied AI — and I am open to internships in foundation-model, AI-Infra and robotics teams.


### Open-source contributions

**[FlagOS](https://github.com/flagos-ai/FlagScale)** — MetaX adaptation, training framework optimization, and bug fixes across FlagScale / Megatron-LM-FL / TransformerEngine-FL

**TransformerEngine-FL**: MetaX TE 2.13+ package layout and loading-path adaptation ([PR #128](https://github.com/flagos-ai/TransformerEngine-FL/pull/128))<br>
**FlagScale**: MetaX argument-check and profiler fixes ([PR #1296](https://github.com/flagos-ai/FlagScale/pull/1296))<br>
**Megatron-LM-FL**: MetaX JIT/Dynamo memory-leak fix ([PR #185](https://github.com/flagos-ai/Megatron-LM-FL/pull/185))<br>
**Chunk CE**: Chunked vocabulary-parallel cross-entropy to reduce temporary training memory ([PR #126](https://github.com/flagos-ai/Megatron-LM-FL/pull/126))<br>
**TP overlap**: Communication–computation overlap adaptation and fixes on MetaX ([PR #139](https://github.com/flagos-ai/TransformerEngine-FL/pull/139))<br>
**FlagScale multimodal packing**: BFD / Next Fit packing algorithms; Qwen3.5 Next Fit with multimodal alignment and per-sample position encoding ([PR #1308](https://github.com/flagos-ai/FlagScale/pull/1308))<br>

**[Hugging Face](https://github.com/huggingface/peft)**

**PEFT / LoRA-FA**: Optimizer `step(closure)` gradient-context fix and regression test ([PR #3966](https://github.com/huggingface/peft/pull/3966))
