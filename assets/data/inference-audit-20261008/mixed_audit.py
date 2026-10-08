"""Bounded synthetic stream audit; run only on an allocated NPU.

This measures synthetic operators, not serving or real KV attention.
Example: python mixed_audit.py --device 1 --steps 30 --repeats 5
"""
import argparse
import json
import random
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--prefill-multiple', type=int, default=3)
    args = parser.parse_args()
    if args.device == 0 or min(args.steps, args.repeats, args.prefill_multiple) < 1:
        parser.error('device 0 is excluded; counts must be positive')

    import torch
    import torch_npu

    torch.manual_seed(42)
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    x = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    w = torch.randn_like(x)
    k = torch.randn(16384, 8192, dtype=torch.float16, device=device)
    q = torch.randn(1, 8192, dtype=torch.float16, device=device)
    prefill_stream, decode_stream = torch.npu.Stream(), torch.npu.Stream()

    def prefill():
        return torch.softmax((x @ w).float(), dim=-1)

    def decode():
        return torch.softmax((q @ k.transpose(0, 1)).float(), dim=-1)

    for _ in range(10):
        prefill()
        decode()
    torch.npu.synchronize()

    def trial(mode, repeat):
        torch.npu.synchronize()
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        wall_start = time.perf_counter()

        def enqueue_prefill():
            with torch.npu.stream(prefill_stream):
                for _ in range(args.steps * args.prefill_multiple):
                    prefill()

        def enqueue_decode():
            with torch.npu.stream(decode_stream):
                begin.record()
                for _ in range(args.steps):
                    decode()
                end.record()

        if mode == 'prefill_only':
            enqueue_prefill()
            submit_end = time.perf_counter()
            prefill_stream.synchronize()
            print(json.dumps({'mode': mode, 'repeat': repeat,
                              'submit_ms': (submit_end-wall_start)*1000,
                              'prefill_completion_ms': (time.perf_counter()-wall_start)*1000}), flush=True)
            return
        if mode in ('prefill_first', 'serial'):
            enqueue_prefill()
        if mode == 'serial':
            prefill_stream.synchronize()
        decode_submit_start = time.perf_counter()
        if mode == 'alternating':
            with torch.npu.stream(decode_stream):
                begin.record()
            for _ in range(args.steps):
                with torch.npu.stream(prefill_stream):
                    for _ in range(args.prefill_multiple):
                        prefill()
                with torch.npu.stream(decode_stream):
                    decode()
            with torch.npu.stream(decode_stream):
                end.record()
        else:
            enqueue_decode()
        decode_submit_end = time.perf_counter()
        if mode == 'decode_first':
            enqueue_prefill()
        end.synchronize()  # Decode completion only, not whole-device completion.
        decode_done = time.perf_counter()
        device_span_ms = begin.elapsed_time(end)
        torch.npu.synchronize()  # Cleanup is outside the decode completion timestamp.
        all_done = time.perf_counter()
        print(json.dumps({
            'mode': mode, 'repeat': repeat, 'steps': args.steps,
            'decode_device_span_ms': device_span_ms,
            'decode_device_span_per_step_ms': device_span_ms/args.steps,
            'decode_submission_ms': (decode_submit_end-decode_submit_start)*1000,
            'decode_completion_from_submit_start_ms': (decode_done-decode_submit_start)*1000,
            'decode_completion_from_trial_start_ms': (decode_done-wall_start)*1000,
            'whole_trial_ms': (all_done-wall_start)*1000,
        }), flush=True)

    print(json.dumps({'metadata': vars(args), 'torch': torch.__version__,
                      'torch_npu': torch_npu.__version__,
                      'note': 'Event span includes execution/queue gaps; it is not sum of kernel durations.'}), flush=True)
    rng = random.Random(42)
    for repeat in range(args.repeats):
        modes = ['decode_only', 'prefill_only', 'serial', 'prefill_first', 'decode_first', 'alternating']
        rng.shuffle(modes)
        for mode in modes:
            trial(mode, repeat)


if __name__ == '__main__':
    main()
