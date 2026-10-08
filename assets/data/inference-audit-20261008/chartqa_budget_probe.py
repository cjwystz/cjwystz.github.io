"""Diagnostic real-image, quality/cost probe; HF eager runner, not a serving claim.

ChartQA relaxed accuracy follows the official evaluator: case-insensitive exact
match, or <=5% relative numerical error. Small deterministic stratified sample.
"""
import argparse
import hashlib
import io
import json
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
import torch
import torch_npu
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def relaxed_correct(prediction, target):
    prediction, target = prediction.strip().lower(), target.strip().lower()
    try:
        def number(s):
            return float(s[:-1]) / 100 if s.endswith('%') else float(s)
        p, t = number(prediction), number(target)
        return p == t if t == 0 else abs(p - t) / abs(t) <= 0.05
    except ValueError:
        return prediction == target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/models/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--dataset', default='/data/chartqa_test.parquet')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--budgets', type=int, nargs='+', default=[64, 256, 1024, 4096])
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--attn', default='sdpa')
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(a.dataset).to_pylist()
    rng = random.Random(20261008)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['type'], []).append(i)
    indices = []
    for group in sorted(groups):
        rng.shuffle(groups[group])
        indices.extend(groups[group][:a.samples // len(groups)])
    rng.shuffle(indices)
    meta = {'model': a.model, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'transformers': transformers.__version__, 'seed': 20261008,
            'dataset_sha256': hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
            'dataset_rows': len(rows), 'groups': {k: len(v) for k, v in groups.items()},
            'sample_indices': indices, 'budgets': a.budgets, 'max_new_tokens': a.max_new_tokens,
            'metric': 'ChartQA official relaxed accuracy (5% relative numerical tolerance)',
            'runner': 'HF single request; cold warmup reported separately; no server batching'}
    out.with_suffix('.manifest.json').write_text(json.dumps(meta, indent=2))
    row = rows[indices[0]]
    Image.open(io.BytesIO(row['image']['bytes'])).save(out.with_suffix('.sample.png'))
    print(json.dumps({'sample_index': indices[0], 'question': row['question'], 'answer': row['answer']}), flush=True)
    if a.prepare_only:
        return
    torch.npu.set_device(0)  # own isolated container maps physical card 3
    torch.npu.config.allow_internal_format = False
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation=a.attn, local_files_only=True).eval().to('npu')
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    visual = model.model.visual
    print(json.dumps({'vision_config': model.config.vision_config.to_dict(), 'attention': a.attn}), flush=True)
    measurements = {}
    def start_forward(module, args, kwargs):
        if 'first_start' not in measurements:
            measurements['first_start'] = torch.npu.Event(enable_timing=True)
            measurements['first_start'].record()
    def end_forward(module, args, kwargs, result):
        if 'first_end' not in measurements:
            measurements['first_end'] = torch.npu.Event(enable_timing=True)
            measurements['first_end'].record()
            torch.npu.synchronize()
            measurements['first_wall'] = time.perf_counter()
    def start_visual(module, args):
        measurements['vision_start'] = torch.npu.Event(enable_timing=True)
        measurements['vision_start'].record()
    def end_visual(module, args, result):
        measurements['vision_end'] = torch.npu.Event(enable_timing=True)
        measurements['vision_end'].record()
    model.register_forward_pre_hook(start_forward, with_kwargs=True)
    model.register_forward_hook(end_forward, with_kwargs=True)
    visual.register_forward_pre_hook(start_visual)
    visual.register_forward_hook(end_visual)
    def run(index, budget, warmup=False):
        row = rows[index]
        image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        prompt = row['question'] + '\nAnswer with the answer only, without explanation.'
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        measurements.clear()
        start = time.perf_counter()
        inputs = processor(text=[text], images=[image], return_tensors='pt',
                           min_pixels=4 * 28 * 28, max_pixels=budget * 28 * 28)
        processor_ms = (time.perf_counter() - start) * 1000
        grid = inputs.image_grid_thw.tolist()
        input_length = inputs.input_ids.shape[1]
        inputs = inputs.to('npu')
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens,
                                       use_cache=True, logits_to_keep=1)
        torch.npu.synchronize()
        finish = time.perf_counter()
        output_ids = generated[0, input_length:].tolist()
        answer = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        result = {'index': index, 'type': row['type'], 'question': row['question'], 'target': row['answer'],
                  'prediction': answer, 'correct': relaxed_correct(answer, row['answer']),
                  'budget': budget, 'image_size': list(image.size), 'grid_thw': grid,
                  'visual_tokens': sum(t*h*w//4 for t,h,w in grid), 'input_tokens': input_length,
                  'output_tokens': len(output_ids), 'output_ids': output_ids, 'warmup': warmup,
                  'processor_ms': processor_ms, 'ttft_wall_ms': (measurements['first_wall']-start)*1000,
                  'first_forward_npu_ms': measurements['first_start'].elapsed_time(measurements['first_end']),
                  'vision_npu_ms': measurements['vision_start'].elapsed_time(measurements['vision_end']),
                  'total_wall_ms': (finish-start)*1000}
        return result
    with out.open('w') as handle:
        warmup = run(indices[0], min(a.budgets), True)
        handle.write(json.dumps(warmup) + '\n'); handle.flush()
        print(json.dumps(warmup), flush=True)
        jobs = [(index,budget) for index in indices for budget in a.budgets]
        rng.shuffle(jobs)
        for index, budget in jobs:
            result = run(index, budget)
            handle.write(json.dumps(result) + '\n'); handle.flush()
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
