"""Replay fixed input-ID arrivals and retain cumulative token stream events.

SLO thresholds are explicit inputs, not chosen from achieved results. Random input
IDs are a synthetic shape workload; they are not a representative language trace.
"""
import argparse
import asyncio
import json
import pathlib
import random
import statistics
import time


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[round((len(xs)-1)*q)]


async def main(a):
    import aiohttp
    trace = json.loads(pathlib.Path(a.trace).read_text())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=300)) as session:
        async def request(spec, epoch, warmup=False):
            due = epoch + spec['arrival_s']
            await asyncio.sleep(max(0, due-time.perf_counter()))
            started = time.perf_counter()
            rng = random.Random(spec['prompt_seed'])
            ids = [rng.randrange(1000, 12000) for _ in range(spec['input_tokens_target'])]
            payload = {'input_ids': ids, 'sampling_params': {
                'temperature': 0, 'max_new_tokens': spec['output_tokens_target'], 'ignore_eos': True},
                'stream': True}
            events = []
            status = None
            last_count = 0
            final = {}
            error = None
            try:
                async with session.post(a.url+'/generate', json=payload) as response:
                    status = response.status
                    if status != 200:
                        raise RuntimeError((await response.text())[:500])
                    async for raw in response.content:
                        line = raw.strip()
                        if not line.startswith(b'data:') or line == b'data: [DONE]':
                            continue
                        data = json.loads(line[5:].strip())
                        if 'error' in data:
                            raise RuntimeError(str(data['error']))
                        final = data
                        count = data.get('meta_info', {}).get('completion_tokens', 0)
                        if count > last_count:
                            events.append({'time_s': time.perf_counter()-started,
                                           'completion_tokens': count, 'delta': count-last_count})
                            last_count = count
            except Exception as exc:
                error = str(exc)
            finished = time.perf_counter()
            meta = final.get('meta_info', {})
            row = {'request_id': spec['request_id'], 'input_tokens_target': len(ids),
                   'output_tokens_target': spec['output_tokens_target'],
                   'scheduled_arrival_s': spec['arrival_s'],
                   'actual_arrival_s': started-epoch, 'dispatch_lag_s': started-due,
                   'http_status': status, 'error': error, 'events': events,
                   'actual_prompt_tokens': meta.get('prompt_tokens'),
                   'actual_completion_tokens': meta.get('completion_tokens'),
                   'e2e_s': finished-started, 'warmup': warmup}
            row['ttft_s'] = events[0]['time_s'] if events else None
            # Last token arrival, excluding HTTP teardown and counting real tokens.
            row['tpot_s'] = ((events[-1]['time_s']-events[0]['time_s'])/(last_count-1)
                             if events and last_count > 1 else None)
            row['itl_s'] = ([events[i]['time_s']-events[i-1]['time_s']
                              for i in range(1,len(events))]
                             if events and all(e['delta']==1 for e in events) else None)
            if events and any(e['delta'] != 1 for e in events):
                row['coalesced_stream'] = True
            return row

        warm_spec = {'request_id': -1, 'arrival_s': 0, 'prompt_seed': 0,
                     'input_tokens_target': 512, 'output_tokens_target': 16}
        warm = await request(warm_spec, time.perf_counter(), True)
        if warm['error'] or warm['actual_completion_tokens'] != 16:
            raise RuntimeError('warmup failed: '+json.dumps(warm))
        epoch = time.perf_counter()
        rows = await asyncio.gather(*(request(spec, epoch) for spec in trace['requests']))
        elapsed = time.perf_counter()-epoch
    ok = [r for r in rows if r['error'] is None and r['actual_completion_tokens']==r['output_tokens_target']]
    valid = [r for r in ok if r['tpot_s'] is not None]
    slo = [r for r in valid if r['ttft_s'] <= a.ttft_slo and r['tpot_s'] <= a.tpot_slo]
    summary = {'config': vars(a), 'trace_config': trace['config'], 'requests': len(rows),
               'completed_exact_length': len(ok), 'failed_or_short': len(rows)-len(ok),
               'arrival_window_s': trace['config']['duration'], 'completion_window_s': elapsed,
               'slo_satisfied': len(slo),
               'slo_goodput_arrival_window_rps': len(slo)/trace['config']['duration'],
               'slo_goodput_including_drain_rps': len(slo)/elapsed,
               'output_tok_s_including_drain': sum(r['actual_completion_tokens'] for r in ok)/elapsed,
               'ttft_p50_s': pct([r['ttft_s'] for r in ok if r['ttft_s'] is not None],.5),
               'ttft_p99_s': pct([r['ttft_s'] for r in ok if r['ttft_s'] is not None],.99),
               'tpot_p50_s': pct([r['tpot_s'] for r in valid],.5),
               'tpot_p99_s': pct([r['tpot_s'] for r in valid],.99),
               'itl_p99_s': pct([x for r in valid if r['itl_s'] is not None for x in r['itl_s']],.99),
               'coalesced_requests': sum(bool(r.get('coalesced_stream')) for r in rows),
               'max_dispatch_lag_s': max(r['dispatch_lag_s'] for r in rows)}
    destination = pathlib.Path(a.output)
    destination.write_text(json.dumps({'summary': summary, 'requests': rows},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:31983')
    p.add_argument('--trace', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ttft-slo', type=float, default=1.0)
    p.add_argument('--tpot-slo', type=float, default=0.05)
    asyncio.run(main(p.parse_args()))
