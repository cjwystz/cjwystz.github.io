"""Controlled fused-attention versus independent copy/GEMM interference probe.

Copy targets never alias active KV. This isolates execution contention, not
correctness hazards or complete PD serving. Raw event pairs are retained.
"""
import argparse
import contextlib
import json
import math
import random
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--seq', type=int, default=8192)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--kv-heads', type=int, default=32)
    p.add_argument('--attention', choices=['fusion', 'inference'], default='fusion')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--copy-mb', type=int, default=64)
    p.add_argument('--copy-count', type=int, default=100)
    p.add_argument('--competitor', choices=['h2d', 'd2d', 'gemm', 'gemm_softmax'], default='h2d')
    p.add_argument('--priority', type=int, default=0)
    p.add_argument('--decode-cube', type=int, default=-1)
    p.add_argument('--decode-vector', type=int, default=-1)
    p.add_argument('--competitor-cube', type=int, default=-1)
    p.add_argument('--competitor-vector', type=int, default=-1)
    p.add_argument('--profile-dir')
    a = p.parse_args()
    if a.device == 0:
        p.error('physical card 0 is excluded')
    import torch
    import torch_npu
    torch.manual_seed(42)
    torch.npu.set_device(a.device)
    dev = f'npu:{a.device}'
    q = torch.randn(a.batch, a.heads, 1, 128, dtype=torch.float16, device=dev)
    k = torch.randn(a.batch, a.kv_heads, a.seq, 128, dtype=torch.float16, device=dev)
    v = torch.randn_like(k)
    copy_elems = a.copy_mb * 1024 * 1024 // 2
    if a.competitor == 'h2d':
        source = torch.empty(copy_elems, dtype=torch.float16, pin_memory=True).fill_(0.5)
    else:
        source = torch.full((copy_elems,), 0.5, dtype=torch.float16, device=dev)
    dest = torch.empty(copy_elems, dtype=torch.float16, device=dev)
    x = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    w = torch.randn_like(x)
    ds = torch.npu.Stream(priority=a.priority)
    cs = torch.npu.Stream()
    if a.decode_cube != -1 or a.decode_vector != -1:
        torch.npu.set_stream_limit(ds, cube_num=a.decode_cube, vector_num=a.decode_vector)
    if a.competitor_cube != -1 or a.competitor_vector != -1:
        torch.npu.set_stream_limit(cs, cube_num=a.competitor_cube, vector_num=a.competitor_vector)

    def attn():
        if a.attention == 'inference':
            return torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=a.heads, num_key_value_heads=a.kv_heads,
                input_layout='BNSD', scale=1/math.sqrt(128),
                actual_seq_lengths_kv=[a.seq]*a.batch)[0]
        return torch_npu.npu_fusion_attention(q, k, v, a.heads, 'BNSD',
                                             scale=1/math.sqrt(128), keep_prob=1.0)[0]

    def competitor():
        if a.competitor in ('h2d', 'd2d'):
            dest.copy_(source, non_blocking=True)
        elif a.competitor == 'gemm':
            return x @ w
        else:
            return torch.softmax((x @ w).float(), dim=-1)

    for _ in range(10):
        attn()
        competitor()
    torch.npu.synchronize()
    # Sample every batch's first query head against an FP32 CPU reference.
    # This head maps to KV head 0 under GQA; validate under the actual quota.
    qc = q[:, :1].float().cpu()
    kc = k[:, :1].float().cpu()
    vc = v[:, :1].float().cpu()
    reference = torch.softmax((qc @ kc.transpose(-1,-2))/math.sqrt(128),dim=-1) @ vc
    with torch.npu.stream(ds):
        actual = attn()
    ds.synchronize()
    sampled = actual[:, :1].float().cpu()
    validation = {'max_abs_error': (sampled-reference).abs().max().item(),
                  'mean_abs_error': (sampled-reference).abs().mean().item()}
    if not torch.allclose(sampled,reference,atol=0.003,rtol=0.03):
        raise RuntimeError('sampled attention correctness check failed: '+str(validation))
    print(json.dumps({'metadata': vars(a), 'sampled_correctness': validation, 'torch': torch.__version__,
                      'torch_npu': torch_npu.__version__,
                      'decode_stream_limit': torch.npu.get_stream_limit(ds),
                      'competitor_stream_limit': torch.npu.get_stream_limit(cs),
                      'source_pinned': source.is_pinned() if source.device.type == 'cpu' else False}), flush=True)

    def trial(mode, repeat):
        torch.npu.synchronize()
        # Allocate all events before the measured region, for every mode.
        de = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
              for _ in range(a.steps)]
        ce = [torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)]
        t0 = time.perf_counter()

        def enqueue_competitor():
            with torch.npu.stream(cs):
                ce[0].record()
                for _ in range(a.copy_count):
                    competitor()
                ce[1].record()

        def enqueue_decode():
            with torch.npu.stream(ds):
                for begin, end in de:
                    begin.record()
                    attn()
                    end.record()

        if mode in ('competitor_first', 'serial', 'competitor_only'):
            enqueue_competitor()
        if mode == 'serial':
            cs.synchronize()
        submit_start = time.perf_counter()
        if mode != 'competitor_only':
            enqueue_decode()
        submit_end = time.perf_counter()
        if mode == 'decode_first':
            enqueue_competitor()
        if mode != 'competitor_only':
            de[-1][1].synchronize()
        decode_done = time.perf_counter()
        torch.npu.synchronize()
        all_done = time.perf_counter()
        row = {'mode': mode, 'repeat': repeat,
               'decode_submit_ms': (submit_end-submit_start)*1000,
               'decode_completion_from_trial_start_ms': (decode_done-t0)*1000,
               'whole_trial_ms': (all_done-t0)*1000}
        if mode != 'competitor_only':
            row.update({'decode_op_ms': [b.elapsed_time(e) for b,e in de],
                        'decode_start_offsets_ms': [de[0][0].elapsed_time(b) for b,e in de],
                        'decode_span_ms': de[0][0].elapsed_time(de[-1][1])})
        if mode != 'decode_only':
            row['competitor_span_ms'] = ce[0].elapsed_time(ce[1])
        print(json.dumps(row), flush=True)

    cfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    ctx = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True, experimental_config=cfg,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.profile_dir)
    ) if a.profile_dir else contextlib.nullcontext()
    rng = random.Random(42)
    with ctx:
        for repeat in range(a.repeats):
            modes = ['decode_only', 'competitor_only', 'serial', 'competitor_first', 'decode_first']
            rng.shuffle(modes)
            for mode in modes:
                trial(mode, repeat)


if __name__ == '__main__':
    main()
