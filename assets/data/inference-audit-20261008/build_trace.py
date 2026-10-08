"""Generate replayable open-loop arrival traces without coupling prompts to arrivals.

Token IDs must come from the serving tokenizer when benchmarking exact lengths.
This trace stores requested lengths and separate prompt seeds, not tokenized prompts.
"""
import argparse
import json
import random


def build_trace(rate, duration, arrival_seed, prompt_seed, input_lengths, output_lengths):
    arrivals = random.Random(arrival_seed)
    prompts = random.Random(prompt_seed)
    now = 0.0
    rows = []
    while True:
        now += arrivals.expovariate(rate)
        if now >= duration:
            return rows
        rows.append({'request_id': len(rows), 'arrival_s': now,
                     'input_tokens_target': prompts.choice(input_lengths),
                     'output_tokens_target': prompts.choice(output_lengths),
                     'prompt_seed': prompts.getrandbits(64)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rate', type=float, default=8)
    p.add_argument('--duration', type=float, default=90)
    p.add_argument('--arrival-seed', type=int, default=42)
    p.add_argument('--prompt-seed', type=int, default=123)
    p.add_argument('--inputs', type=int, nargs='+', default=[512, 4096])
    p.add_argument('--outputs', type=int, nargs='+', default=[256])
    args = p.parse_args()
    if args.rate <= 0 or args.duration <= 0 or min(args.inputs + args.outputs) <= 0:
        p.error('rate, duration and lengths must be positive')
    rows = build_trace(args.rate, args.duration, args.arrival_seed, args.prompt_seed,
                       args.inputs, args.outputs)
    print(json.dumps({'config': vars(args), 'requests': rows}, indent=2))


if __name__ == '__main__':
    main()
