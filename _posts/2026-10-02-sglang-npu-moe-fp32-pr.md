---
layout: post
title: 追一个静默精度 bug：我给 SGLang 提 PR 的完整记录
date: 2026-10-02 12:00:00 +0800
categories: [engineering]
tags: [sglang, ascend-910b, moe, open-source]
description: 从发现 issue #39351 到提交 PR #42182：如何用真机数值证据证明一行精度修复值得被合入——包括验证过程中踩的坑、两个没修成的前任，以及开源礼仪。
toc:
  beginning: true
---

TL;DR：

- **Bug**：SGLang 昇腾后端在 `AscendTPDispatcher.dispatch()` 里把 MoE 路由权重从 FP32 降成 BF16，而下游 CANN kernel `npu_moe_finalize_routing` 原生接受 FP32——精度是白丢的；
- **影响**：910B3 真机实测，对比 FP64 参考重算，FP32 路径误差 4.0e-7，当前 BF16 路径误差 1.6e-2；
- **PR**：[sgl-project/sglang#42182](https://github.com/sgl-project/sglang/pull/42182)，open 待 review。

## 1. 选目标：怎么挑一个能落地的 issue

SGLang 每天合入几十个 PR，但**外部贡献者能落地的 NPU issue 很少**——瓶颈不是写代码，是验证：昇腾真机不是谁都有。我的筛选标准：

1. 根因被指到具体行（省掉定位时间）；
2. 无人认领、无活跃 PR（避免撞车）；
3. 我手里有它需要的验证条件（910B 真机）。

[issue #39351](https://github.com/sgl-project/sglang/issues/39351) 三条全中：2026-09-14 提出，0 评论，作者把根因直接指到 `ascend_tp.py` 的降精度那一行。

## 2. 这个 bug 是什么意思

MoE 模型每层有 N 个专家，router 给每个 token 选出 top-k 个专家并分配**路由权重**（加和为 1 的小数），最终输出是 k 个专家结果的加权和。权重的精度直接决定"专家意见怎么混合"。

SGLang 里这条链路的契约是：TopK 层输出 FP32 权重（`topk.py` 里 `dtype=torch.float32`），下游 kernel 按 FP32 消费。但昇腾 TP dispatcher 中间插了一行：

```python
topk_weights, topk_ids, _ = topk_output
topk_weights = topk_weights.to(hidden_states.dtype)   # FP32 -> BF16
```

BF16 只有约 3 位有效数字，`0.137` 压过去变成 `0.13672`。单次误差 0.2% 看似无害，但 MoE 有几十层，每层注入一次；而且这是**静默**的——不报错、不崩溃，输出只是悄悄变差。

影响面也确认了：`fused_moe_triton/layer.py` 里 `if a2a_backend.is_none() and is_npu(): return AscendTPDispatcher(...)`——NPU 上默认路径，所有 MoE 模型都经过这行。

## 3. 三段验证：先证明问题，再证明修复

开源礼仪：**先有证据，再认领、再提 PR**。我做了三段验证。

### 3.1 发布版里 bug 真实存在

```bash
SGL=$(python -c "import sglang,os;print(os.path.dirname(sglang.__file__))")
grep -n "topk_weights.to(hidden_states.dtype)" $SGL/srt/layers/moe/token_dispatcher/ascend_tp.py
# 101:        topk_weights = topk_weights.to(hidden_states.dtype)
grep -n "is_none() and is_npu()" $SGL/srt/layers/moe/fused_moe_triton/layer.py
# 139:    if a2a_backend.is_none() and is_npu():
```

注意 grep 的是**已安装的发布版**（v0.5.18），不是仓库源码——证明线上用户正在吃这个亏。

### 3.2 kernel 吃不吃 FP32

最小化调用 `npu_moe_finalize_routing`，分别喂 FP32 / BF16 scales：

```
fp32: OK   out.dtype=torch.bfloat16
bf16: OK   out.dtype=torch.bfloat16
fp32 vs bf16 outputs bitwise identical: False
```

kernel 接受 FP32，且两条路径输出真的不同——降精度不是"无操作"，是在改结果。

### 3.3 决定性一步：FP32 是否更接近真值

"输出不同"不等于"更好"。用 FP64 在 CPU 上重算 finalize 语义作参考答案，比较两条 kernel 路径谁更准：

```python
perm64 = permuted.to("cpu", torch.float64)
w64 = w_fp32.to("cpu", torch.float64)
ri = row_idx.to("cpu", torch.long).view(-1)
# 方向判定：bf16 行拷贝逐位相等，sanity=0 的方向即正确映射
validA = (ri >= 0) & (ri < M)
sanA = (perm64[ri[validA]] - x64[s[validA] // K]).abs().max().item()   # -> 0.0

ref = torch.zeros(T, H, dtype=torch.float64)
ref.index_add_(0, idx // K, perm64[ri[validA]] * w64[idx // K, idx % K].unsqueeze(1))
```

**这里踩了一个坑，值得记下来**：第一版参考实现我假设 permuted 张量按 token 分组排列，结果两条路径误差一模一样（都是 4.88）——参考公式从根上就错了，把真实差异完全淹没。修正方法是加 sanity 检查：permute 是 bf16 行拷贝，逐位相等，所以"permuted 某行是否精确等于 x 某行"能无歧义地判定映射方向（sanity=0 即正确）。修正后：

| 路径 | 与 FP64 参考的最大绝对误差 |
|---|---|
| kernel 喂 FP32 权重 | **4.0e-07** |
| kernel 喂 BF16 权重（当前行为） | **1.6e-02** |
| 纯数学重算 BF16 权重 | 8.0e-03 |

第三行是反证：误差来自**权重降精度本身**，不是 kernel 的怪癖。证据链闭环：契约是 FP32 → 发布版存在违规降精度 → kernel 吃 FP32 且数值精确 → 降精度注入 1.6e-2 量级误差。

## 4. 两个没修成的前任

开 PR 前查竞争情况是必修课。issue 下挂着两个修复 PR，都没合：

- **#39369**：删 cast + 加单测。但作者自述"本地没有 torch，测试没跑过；没有昇腾硬件，没做 NPU 验证"。对 maintainer 来说，合一个没人验证过的硬件后端改动是有风险的，于是两周半零 review；
- **#39394**：只删一行。CI 全红，评论区线程混乱。

"第一个提修复"的名额被占了，但"让修复落地"的名额空着——缺的正是真机验证。我的处理是守礼仪、不装没看见：issue 下先发证据评论说明前情，PR 正文里明确写 `Supersedes #39369 and #39394`，并加一句"如果 maintainer 更愿意把证据合进 #39369，我可以关掉我这个"。**把合作选项摆前面**，开源社区吃这套。

## 5. 环境坑速记（昇腾 docker 开发通用）

在官方 NPU 镜像里做开发，几个坑记一下：

- 容器内是 root，`~` 是 `/root`，仓库实际挂在 `/workspace/...`；
- root 操作宿主机用户的 git 仓库会报 `dubious ownership`：`git config --global --add safe.directory <path>`，并且 `chown` 要用**真实 uid**（域账号环境可能是 1291600093 这种大数，不是 1000）；
- 镜像里**没有 ssh 客户端**（`git push` 报 `cannot run ssh`）：代码在容器里改，commit/push 回宿主机做。

## 6. 总结

- 修复本身是一行：`topk_weights = topk_weights.to(torch.float32)`。**代码不值钱，证据值钱**——maintainer 要的不是"有人改了"，是"有人证明了改是对的、且不会改坏别的"；
- 静默精度 bug 是最难被发现的一类：不报错、不崩溃、结果悄悄变差。抓它需要真机 + 愿意逐行读 dispatcher 的耐心；
- 国产芯片生态的开源贡献，稀缺资源是**硬件访问权**。有真机的人把验证做扎实，就是别人替代不了的贡献；
- PR 当前状态：open，CI 门禁等 maintainer 打 `run-ci` 标签后真跑。无论最终是合我这个、还是把证据并进前任的 PR，这张误差表都会留在公开记录里。

相关：[issue #39351](https://github.com/sgl-project/sglang/issues/39351) · [PR #42182](https://github.com/sgl-project/sglang/pull/42182) · 同系列：[昇腾 910B 部署 SGLang 推理服务实战](/blog/2026/sglang-ascend-910b-deploy/)
