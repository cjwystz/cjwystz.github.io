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
    box-shadow: 0 8px 30px rgba(59, 91, 219, 0.25);
  }
</style>

I am an **AI Infra researcher & engineer** focused on **large-model distributed training and inference systems**, memory optimization, and load scheduling. I am an undergraduate (2023–2027) in Information Security at the **University of Science & Technology Beijing**, and an AI-Infra intern at **Qingcheng Jizhi**, contributing to the FlagOS open-source ecosystem.

My hands-on experience spans **1000-GPU-scale training and inference across heterogeneous accelerators** — Huawei Ascend 910C, MetaX C550, and Ali Zhenwu 810E — alongside NVIDIA A800/A100 clusters. My research on heterogeneous-modeling Packing scheduling (MLSys 2027) delivers **+20% end-to-end training throughput**, and my work on formal-proof-guided geometric reasoning (TMLR 2027, under review) improves GRPO-trained reasoning by **+9.7%** on geometry and **+8.0%** on general math.

I am actively seeking a **PhD (2028–2029 Fall) in large-model systems** — distributed training/inference, memory & scheduling optimization, and embodied AI — and I am open to internships in foundation-model, AI-Infra and robotics teams.
