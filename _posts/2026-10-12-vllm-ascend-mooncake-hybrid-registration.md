---
layout: post
title: "给 vLLM-Ascend 修一个 KV 缓存注册 bug：从 tensor view 到真实 P/D 验证"
date: 2026-10-12 09:00:00 +0800
categories: [engineering, llm-serving]
tags: [vllm-ascend, mooncake, ascend-910b3, kv-cache, disaggregation, open-source]
description: 记录 Mooncake V1 混合注意力注册路径的一个小修复：恢复底层存储地址、合并重叠区域，并用 CPU 回归与真实 Gemma 3 双卡 P/D 验证明确证据边界。
toc:
  beginning: true
---

这次给 vLLM-Ascend 提交了一个范围很小的修复：[PR #18355](https://github.com/vllm-project/vllm-ascend/pull/18355)。它处理 Mooncake V1 在混合注意力模型中注册 KV 缓存时的地址问题：逻辑 tensor view 的起点，不一定是底层内存应该注册的起点。

生产代码的修改只有新增 4 行、删除 18 行，复用了仓库已有的存储感知注册函数。验证包括四个 CPU 回归用例，以及真实 Gemma 3 1B 的双卡 Prefill/Decode（P/D）功能测试。三个提示词下，P/D 生成的 token ID 都与直接生成一致。

截至 2026 年 10 月 12 日，PR 已转为 Ready for review，并邀请维护者审查，**尚未合并**。本文记录修复和验证过程，不提出服务吞吐提升的结论。

## 为什么 tensor 地址不等于注册地址

P/D 分离把提示词计算放到 Prefill 实例，把后续生成放到 Decode 实例。Prefill 产生的 KV 缓存需要交给 Decode。使用 Mooncake 时，传输后端首先要注册能够访问的内存区域，再根据各层的地址、block 长度和 stride 传输数据。

PyTorch 的 tensor 可以是底层 storage 的一个 view。假设有一块 8 MiB 的内存，起点已经满足 2 MiB 对齐，但某层的 view 从起点后 128 字节开始：

```text
底层内存： [base ................................ base + 8 MiB)
层的 view：        [base + 128 .................. base + 8 MiB)
```

`view.data_ptr()` 指向 `base + 128`。这个地址用于定位层的数据是正确的，但如果把它作为整个底层内存的注册起点，就会出现问题：它没有满足 2 MiB 对齐；从这个起点再注册完整的 8 MiB，还可能超出实际分配边界。

另一种情况是两个缓存描述符共享同一块 storage。分别以各自 view 地址和描述符大小注册，可能产生重复或重叠范围。即使两个起点恰好都对齐，注册范围仍可能不正确。

因此需要分别保存两类信息：**底层内存的注册范围，以及各层数据的逻辑地址。** 前者交给传输后端，后者用于计算具体要传输的数据位置。

## 修复的是哪条分支

问题位于 `mooncake_connector.py` 中的 `_get_registered_kv_tensor_buffers_hybrid()`。原来的实现收集各层 view 的地址，取最小值，再按描述符大小注册，并检查这个起点是否满足 2 MiB 对齐。

仓库已经有 `_get_registered_kv_tensor_buffers()`，用于从真实 storage 恢复对齐起点、处理独立层内存，并合并重叠范围。这次让混合注意力分支复用它：

```python
def _get_registered_kv_tensor_buffers_hybrid(self, kv_caches):
    return self._get_registered_kv_tensor_buffers(kv_caches)
```

上面省略了类型标注和解释性注释，核心行为就是这次委托调用。底层恢复和合并算法本来就存在，并非这个 PR 新实现的算法。

在 `register_kv_caches()` 中，原有的选择顺序是：先处理 Mamba 组，再处理 `use_hybrid` 分支，最后处理普通注意力缓存。本次改变的是没有进入 Mamba 分支的混合注意力路径。各层的 `data_ptr()`、block 长度、stride 和 scale 仍由原来的元数据逻辑保存，不会因为注册范围合并而被替换成底层 storage 起点。

可以直接对照[生产代码](https://github.com/vllm-project/vllm-ascend/blob/880f630e4fee7d2fe2a52423e3349a9b42f2d86f/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L2583)和 [PR diff](https://github.com/vllm-project/vllm-ascend/pull/18355/files)。这个修改没有新增 API 或配置，也没有改变注意力计算或调度算法。

## 先证明旧实现会失败

新增测试共四个执行用例，其中一个测试函数参数化为两个场景：

| 场景 | 检查内容 |
| --- | --- |
| view 从底层起点后 128 字节开始 | 恢复对齐的底层起点，逻辑偏移保持不变 |
| 两个描述符共享 storage，view 分别从起点和后 2 MiB 开始 | 最终只注册一次实际底层范围 |
| 一个描述符涉及两块独立层内存 | 分别注册实际内存，不能误合成一块 |
| 从 `register_kv_caches()` 入口注册 | 后端收到底层范围，而层地址、block 长度和 stride 保持逻辑语义 |

在未修改的 Ascend main 提交 `8c0667e52686602726e97972f9b19704d20d4411` 上，四个用例全部失败。应用补丁后，完整 Mooncake V1 单元测试文件的结果是：

```text
192 passed, 14 skipped
```

14 个跳过项是仓库原有的测试，没有为了让这个 PR 通过而新增跳过。这也意味着更广泛的物理存储注册问题还没有全部解决。

CPU 测试使用真实 PyTorch storage 和指定版本的 vLLM Python 代码，加速器及平台启动边界使用仓库已有的 mock。入口测试也没有启动真实传输后端，因此它能验证地址与元数据契约，还需要真机测试补充运行证据。

测试命令为：

```bash
python -m pytest -q tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py
```

## 用真实混合注意力模型跑 P/D

Qwen3.5 虽然也是混合架构，但带有 Mamba 类型缓存，会先进入另一条注册分支。为了覆盖本次修改，选择了 **Gemma 3 1B IT**：它同时使用滑动窗口注意力与全注意力，没有 Mamba 组。

测试使用两张 Ascend 910B3，Prefill 和 Decode 各占一张卡，各自 TP=1。使用原始真实权重；下载文件按模型仓库提供的 SHA256 校验，没有裁剪层数或使用随机权重。

主要配置如下：

| 项目 | 配置 |
| --- | --- |
| 计算模式 | BF16、eager，关闭图编译 |
| 随机种子与采样 | seed=42，temperature=0 |
| 最大模型长度／批处理 token 数 | 2048／2048 |
| block 大小 | 128 token |
| 每个引擎的 KV 内存预算 | 256 MiB |
| 前缀缓存 | 关闭，避免本地缓存命中掩盖远程传输 |
| Connector | MooncakeConnectorV1，分别配置 producer／consumer |

两个实例启动后，日志都记录了六个 `SlidingWindowSpec` 组和一个 `FullAttentionSpec` 组，并完成真实内存注册。组数由实际 KV 缓存分组产生，不等于注意力类型的数量。这与源码中的分支条件一起确认了混合注意力注册路径被执行。

每个用例先在 Decode 实例上直接生成作为参考，再让 Prefill 计算提示词、返回远程 KV 元数据，最后由 Decode 读取这些 KV 并生成。比较的是生成的 **token ID**，同时检查实际传输日志，避免只看到 HTTP 返回成功就认为 P/D 已经生效。

| 提示词 token 数 | 两种路径一致的生成 token 数 | 成功 KV 传输量 |
| --- | --- | --- |
| 6 | 32 | 3,407,872 字节（3.25 MiB） |
| 14 | 32 | 3,407,872 字节（3.25 MiB） |
| 1088 | 32 | 19,136,512 字节（18.25 MiB） |

三个用例全部通过。1088-token 的输入超过了模型 512-token 的滑动窗口。最终运行没有使用注册或传输 mock，日志记录了有实际长度的 Mooncake/HCCL 跨卡传输与完成事件。

这些数据只用于证明功能和传输路径有效。小提示词也传输数 MiB，是因为检查的是按 block 和各层组织的实际 KV 范围，不能拿这张表推导传输效率或服务性能。

## 环境问题也必须留在记录里

这次真机验证并不是第一次启动就成功。有几处失败来自测试环境搭建，不能把它们归因于 PR，也不能从报告里抹掉。

首先，测试追踪代码在 Python 启动时过早导入模块，导致模型结构检查子进程异常退出。去掉追踪后，结构检查通过；最终测试使用未插入追踪的生产路径。

其次，启动脚本曾覆盖镜像原有的 `PYTHONPATH`，让 worker 找不到 CANN 的 `acl` 模块。保留原有路径后解决。当前 worker 还需要仓库 CI 指定的 `fla_npu`，这项依赖安装在隔离环境中。

更容易误判的是 `aclnnAddRmsNormBias`。加载模型后提示找不到这个算子接口，初步看起来像镜像不兼容。但继续检查发现：**完整算子包本来就在镜像中，是搭建环境时只复用了扩展 `.so`，遗漏了配套 `_cann_ops_custom` 包及搜索路径。** 补上 `ASCEND_CUSTOM_OPP_PATH` 和库路径后，真实 NPU 算子调用成功，无需重新下载或编译。

最后，测试脚本为了限制等待时间，把 `HCCL_CONNECT_TIMEOUT` 设成了 60 秒。真实连接时 HCCL 明确报错，要求至少 120 秒。改回部署指南使用的连接超时 120 秒、执行超时 204 秒后，跨卡传输成功。

这些调整都只发生在隔离测试环境，没有加入生产代码 diff，也没有通过 mock 或回退策略绕过传输失败。

## 验证边界与审查状态

Python 源码使用本 PR 的修改和仓库指定的 vLLM 提交 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。底层运行环境复用已有 `v0.26.0rc1` 镜像，包含 CANN 9.1、torch-npu 2.10 及完整原生算子包，没有重新构建当前 main 的全部原生组件。这个来源和限制已经写入 PR 说明。

真实 P/D 测试证明了修改后的路径可以完成混合模型注册、跨卡传输和一致的生成，但**没有在真实模型上做旧代码与新代码的成对 bug 复现**。旧实现会失败的证据来自前面的四个 CPU 回归用例。也没有验证多节点、图模式、并发负载或性能提升。

本 PR 与 [issue #16632](https://github.com/vllm-project/vllm-ascend/issues/16632) 相关，但只处理剩余的 V1 混合注意力分支，没有声明关闭整个 issue。[PR #16129](https://github.com/vllm-project/vllm-ascend/pull/16129) 修改的是相邻的 `mooncake_hybrid_connector.py`，本次修改的是 V1 的 `mooncake_connector.py`，两者范围不同。

截至本文发布，pre-commit、DCO 和针对 verified vLLM main／v0.30.0 的两组 CPU UT 已通过。仓库的最终 CI gate 还要求维护者添加测试选择标签，安排精确测试或完整测试；因此不能把当前状态写成“所有合并检查已经通过”。

PR 已进入审查阶段，已礼貌邀请代码负责人和相关维护者提供反馈。后续是否接受这条复用路径、还需要哪些验证，以维护者审查为准。测试完成后，使用的两张卡已释放，原始日志和失败记录保留用于复查。

## 参考

- [PR #18355：代码、测试与审查记录](https://github.com/vllm-project/vllm-ascend/pull/18355)
- [vLLM-Ascend 贡献指南](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)
- [测试与 PR 精确测试说明](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/testing.html)
- [Gemma 3 1B IT 模型仓库](https://modelscope.cn/models/LLM-Research/gemma-3-1b-it)
- [本次使用的 verified vLLM 提交](https://github.com/vllm-project/vllm/commit/ced6857afa0ea7b2e3f0846a62e1394e90f15607)
