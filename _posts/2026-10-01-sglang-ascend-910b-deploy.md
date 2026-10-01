---
layout: post
title: 昇腾 910B 部署 SGLang 推理服务实战：从容器到可用 API
date: 2026-10-01 10:00:00 +0800
categories: [engineering]
tags: [ascend-910b, sglang, inference, ai-infra]
description: 在 Atlas 800I A2（8×910B3）上用官方 NPU 镜像部署 SGLang 推理服务的完整记录——容器挂载清单、设备可见性排查、多卡隔离、无外网权重方案与逐条验证命令。
toc:
  beginning: true
---

本文记录在昇腾 910B 服务器上从零部署 SGLang 推理服务的完整流程。每一步都给出命令、预期输出，以及我实际踩过的坑和排查过程。

## 0. 环境与目标

- **机器**：Atlas 800I A2，8×910B3，驱动 25.5.1
- **镜像**：`quay.io/ascend/sglang:v0.5.18-cann9.0.0-910b`（CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0 / sglang 0.5.18）
- **模型**：Qwen3-8B，本地权重（集群无外网，无法直连 HuggingFace）
- **目标**：起一个可通过 HTTP 访问的推理服务

两个前提约束先交代清楚：

1. **本机 4 号卡已被其他任务的容器长期占用**，本次只用 0–3 号卡；
2. **集群无外网**，`huggingface.co` 和 `hf-mirror.com` 均不通，只有 ModelScope 可直连。所以模型必须走本地权重或 ModelScope（见第 6 节）。

## 1. 部署前检查

### 1.1 确认 NPU 健康状态

```bash
npu-smi info
```

预期输出一张表，8 张 910B3 的 `Health` 列均为 `OK`。同时看一眼每行的显存占用（`HBM-Usage`）——我就是从这里确认 4 号卡已有任务在跑，从而决定只用 0–3 号卡。

### 1.2 确认镜像已在本地

无外网集群拉不了镜像，确认镜像已提前拉好：

```bash
docker images | grep sglang
# quay.io/ascend/sglang   v0.5.18-cann9.0.0-910b   ...
```

### 1.3 软件栈版本（容器内确认）

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
python -c "import torch, torch_npu, sglang; \
  print(torch.__version__, torch_npu.__version__, sglang.__version__)"
# 2.10.0+cpu 2.10.0 0.5.18
```

注意：NPU 镜像是 **CPU 版 torch + torch_npu 插件**的组合，算子实际走 NPU 执行，`torch 2.10.0+cpu` 是正常搭配，不是装错了。

## 2. 启动容器

SGLang 官方 Ascend 文档给出了标准启动命令，按此执行：

```bash
docker run -itd --name sglang-910b \
  --privileged --network=host --ipc=host --shm-size=16g \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc \
  --volume /usr/local/sbin:/usr/local/sbin \
  --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  --volume /etc/ascend_install.info:/etc/ascend_install.info \
  --volume /var/queue_schedule:/var/queue_schedule \
  --volume /data:/data:ro \
  --entrypoint=bash \
  quay.io/ascend/sglang:v0.5.18-cann9.0.0-910b
```

逐参数说明：

| 参数 | 作用 |
|---|---|
| `--privileged` | NPU 设备操作需要特权模式（副作用见第 3 节） |
| `--network=host` | 共享宿主机网络，服务端口直接对外 |
| `--ipc=host --shm-size=16g` | 多进程加载权重需要足够共享内存 |
| `--device=/dev/davinciN` | 暴露 NPU 设备节点 |
| `/dev/davinci_manager`、`/dev/hisi_hdc` | NPU 管理/调试设备，必需 |
| `/usr/local/Ascend/driver`、`firmware` | 驱动与固件目录 |
| `/etc/ascend_install.info` | 驱动安装信息文件 |
| `/usr/local/sbin` | 含 NPU 管理工具脚本 |
| `/var/queue_schedule` | NPU 队列调度目录 |
| `/data:/data:ro` | 本地模型权重，**只读**挂载防误写 |
| `--entrypoint=bash` | 覆盖默认入口，容器起后不自动跑服务 |

**坑 1：挂载不全 → 容器内认不到卡。** 我初次部署时只挂了 `--device` 和 driver 目录，结果容器内 `torch_npu.npu.device_count()` 返回 0。逐一补齐 `ascend_install.info`、`firmware`、`/usr/local/sbin`、`/var/queue_schedule` 后才正常。缺设备和缺这些文件的表现完全一样，都是 device_count=0，排查时不要只盯着 `--device`。

## 3. 验证设备可见性

进容器检查：

```bash
docker exec -it sglang-910b python -c \
  "import torch_npu; print('visible NPUs:', torch_npu.npu.device_count())"
```

输出 `visible NPUs: 8`——**不是 4**。因为 `--privileged` 让容器看见宿主全部 8 张 NPU，`--device` 只挂 4 个并不构成隔离。

**坑 2：多任务混部必须手动做卡级隔离。** 如果不加限制直接起服务，SGLang 会尝试用所有可见的卡，直接撞上 4 号卡上别人的任务。进入容器后第一件事：

```bash
docker exec -it sglang-910b bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

设置后再确认：

```bash
python -c "import torch_npu; print('visible NPUs:', torch_npu.npu.device_count())"
# visible NPUs: 4
```

## 4. 启动推理服务

无外网环境，直接用本地权重。在容器内执行：

```bash
nohup sglang serve \
  --model-path /data/models/Qwen3-8B \
  --attention-backend ascend \
  > sglang_serve.log 2>&1 &

tail -f sglang_serve.log
```

两个参数说明：

- `--model-path`：指向本地模型目录（含 config.json 和权重文件的完整目录）；
- `--attention-backend ascend`：使用昇腾 NPU 的 attention 后端，910B 上必须显式指定。

启动日志关键输出：

```
Capturing batches (bs=1 avail_mem=5.37 GB): 100%|██████| 12/12 [00:41]
max_total_num_tokens=55168, chunked_prefill_size=8192,
max_prefill_tokens=16384, max_running_requests=2048, context_len=40960
Engine startup timings (s): load_weight=4.08, cuda_graph={decode=42.70}
Uvicorn running on http://127.0.0.1:30000
The server is fired up and ready to roll!
```

逐行解读：

- `load_weight=4.08`：权重加载只用了 4 秒；
- `cuda_graph={decode=42.70}`：decode 阶段 graph 捕获 42.7 秒，是**一次性开销**，后续请求不再付出；
- `max_total_num_tokens=55168`：可用 KV cache 容量；
- `context_len=40960`：模型上下文长度；
- 看到 `The server is fired up and ready to roll!` 即启动成功。

默认绑定 `127.0.0.1:30000`。如果需要从其他机器访问，加 `--host 0.0.0.0`；端口用 `--port` 改。

## 5. 服务验证

### 5.1 健康检查

```bash
curl http://127.0.0.1:30000/health
# （200，空响应体即正常）
```

### 5.2 原生 generate 接口

```bash
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 16}
  }'
```

返回：

```json
{"text": " Paris. The capital of Italy is Rome. The capital of Spain is Madrid.",
 "meta_info": {"e2e_latency": 0.311, "completion_tokens": 16}}
```

输出正确，端到端延迟约 0.31s。

### 5.3 OpenAI 兼容接口

SGLang 同时暴露 `/v1/chat/completions`，方便对接现有工具链：

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/Qwen3-8B",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

注意本地权重模式下，`model` 字段填的是模型**路径**而不是模型名。

### 5.4 停止服务

服务跑在容器里，最干净的停止方式：

```bash
docker stop sglang-910b        # 停容器（服务随之终止）
docker start sglang-910b       # 下次重新进容器
```

如果只想停服务保留容器：先 `ps aux | grep sglang` 找到 PID 再 `kill <PID>`。**多人混部的机器上慎用 `pkill -f sglang`**，容易误伤别人的进程。

## 6. 无外网环境的两条路径

如果服务器无法访问 HuggingFace，直接启动会在 server args 解析阶段尝试联网拉 config，抛出一个裸的 `OSError: We couldn't connect to 'https://huggingface.co'...`。这是网络问题不是框架问题，两条绕行路径：

1. **本地权重**（本文方式）：`--model-path` 直接指向已下载的模型目录，全程不联网；
2. **ModelScope**：国内可直连。`export SGLANG_USE_MODELSCOPE=true` 后，`--model-path` 填 ModelScope 模型 ID（如 `Qwen/Qwen3-8B`）即可自动下载。

## 7. 问题速查表

| 现象 | 原因 | 解决 |
|---|---|---|
| `device_count()` 返回 0 | 挂载不全（不只缺 `--device`） | 补齐第 2 节全部 volume |
| 服务想占用全部 8 卡 | `--privileged` 可见所有 NPU | `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` |
| 启动报 `OSError: couldn't connect to huggingface.co` | 无外网，框架尝试在线拉 config | 本地权重或 `SGLANG_USE_MODELSCOPE=true` |
| 起服务慢 | decode graph 捕获 | 正常，一次性开销约 40s |

## 总结

910B 上部署 SGLang 的关键点：

- 容器挂载除 `--device` 外，`ascend_install.info`、firmware、`/usr/local/sbin`、`/var/queue_schedule` 一个都不能少，否则容器内认不到卡；
- `--privileged` 下容器可见全部 NPU，混部场景必须用 `ASCEND_RT_VISIBLE_DEVICES` 做隔离；
- 无外网环境用本地权重或 ModelScope 镜像源即可，报网络错别当成框架 bug；
- 官方镜像 `v0.5.18-cann9.0.0-910b` + Qwen3-8B，从启动到服务可用约 1 分钟（含 graph 捕获），单请求延迟亚秒级。
