"""Compare exact vector-only prototype to vendor IFA, including co-execution."""
import argparse
import contextlib
import json
import math
import random
import statistics
import time
import torch
import torch_npu
from vector_decode import VectorDecode

p=argparse.ArgumentParser()
p.add_argument('--seq',type=int,default=512)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--steps',type=int,default=100)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--profile-dir')
p.add_argument('--graph',action='store_true')
p.add_argument('--quota',action='store_true')
p.add_argument('--grouped',action='store_true')
p.add_argument('--dtype',choices=['fp16','bf16'],default='fp16')
p.add_argument('--competitor',choices=['gemm','ffn'],default='gemm')
p.add_argument('--prefill-tokens',type=int,default=512)
a=p.parse_args()
torch.manual_seed(42)
dtype=torch.float16 if a.dtype=='fp16' else torch.bfloat16
torch.npu.set_device(0) # isolated container maps logical 0 to physical card 3
q=torch.randn(a.batch,32,1,128,device='npu',dtype=dtype)
k=torch.randn(a.batch,8,a.seq,128,device='npu',dtype=dtype)
v=torch.randn_like(k)
x=torch.randn(4096,4096,device='npu',dtype=dtype)
w=torch.randn_like(x)
if a.competitor=='ffn':
    x=torch.randn(a.prefill_tokens,4096,device='npu',dtype=dtype)
    w=torch.randn(4096,24576,device='npu',dtype=dtype)
    down=torch.randn(12288,4096,device='npu',dtype=dtype)
def competitor():
    z=x@w
    if a.competitor=='ffn':
        gate,up=z.chunk(2,dim=-1)
        return (torch.nn.functional.silu(gate)*up)@down
    return z

def vendor():
    return torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=32,num_key_value_heads=8,
        input_layout='BNSD',scale=1/math.sqrt(128),actual_seq_lengths_kv=[a.seq]*a.batch)[0]
variants={'vendor':vendor}
if a.quota:
    for cube in [4,8,12]:variants[f'vendor_reserve{cube}']=vendor
for splits in [4,16]:
    obj=VectorDecode(q,k,splits=splits,block=32)
    variants[f'vector_split{splits}']=lambda obj=obj:obj(q,k,v)
if a.grouped:
    for splits in [4,16]:
        obj=VectorDecode(q,k,splits=splits,block=16,grouped=True)
        variants[f'grouped_split{splits}']=lambda obj=obj:obj(q,k,v)
reference=vendor().clone()
for name,fn in variants.items():
    out=fn()
    torch.npu.synchronize()
    error=(out-reference).abs().max().item()
    if not torch.allclose(out,reference,atol=.003,rtol=.03):
        raise RuntimeError(name+' correctness failure')
    for _ in range(10):fn()
    print(json.dumps({'metadata':vars(a),'variant':name,'max_abs_error_vs_vendor':error}),flush=True)
streams={}
for name in variants:
    ds=torch.npu.Stream()
    cs=torch.npu.Stream()
    if name.startswith('vendor_reserve'):
        cube=int(name.removeprefix('vendor_reserve'))
        torch.npu.set_stream_limit(ds,cube_num=cube,vector_num=2*cube)
        torch.npu.set_stream_limit(cs,cube_num=20-cube,vector_num=8)
    streams[name]=(ds,cs)
ds,cs=streams['vendor']
competitor()
torch.npu.synchronize()
if a.graph:
    graph_variants={}
    captured_outputs=[]
    for name,fn in variants.items():
        ds,cs=streams[name]
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,stream=ds):
            captured_outputs.append(fn())
        graph_variants[name]=graph.replay
        graph.replay()
        torch.npu.synchronize()
        error=(captured_outputs[-1]-reference).abs().max().item()
        if not torch.allclose(captured_outputs[-1],reference,atol=.003,rtol=.03):
            raise RuntimeError(name+' graph correctness failure')
        print(json.dumps({'graph_validation':name,'max_abs_error_vs_vendor':error}),flush=True)
    variants=graph_variants
rng=random.Random(2026)
profile=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.profile_dir)) if a.profile_dir else contextlib.nullcontext()
with profile:
 for repeat in range(a.repeats):
    cases=[(name,mode) for name in variants for mode in ['alone','competitor_first','decode_first']]
    rng.shuffle(cases)
    for name,mode in cases:
        ds,cs=streams[name]
        torch.npu.synchronize()
        events=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(a.steps)]
        ce=[torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)]
        start=time.perf_counter()
        def compete():
            with torch.npu.stream(cs):
                ce[0].record()
                for _ in range(100): competitor()
                ce[1].record()
        if mode=='competitor_first':compete()
        with torch.npu.stream(ds):
            for begin,end in events:
                begin.record();variants[name]();end.record()
        if mode=='decode_first':compete()
        events[-1][1].synchronize()
        decode_done=time.perf_counter()
        torch.npu.synchronize()
        row={'variant':name,'mode':mode,'repeat':repeat,
             'decode_stream_limit':torch.npu.get_stream_limit(ds),'competitor_stream_limit':torch.npu.get_stream_limit(cs),
             'op_ms':[b.elapsed_time(e) for b,e in events],
             'decode_span_ms':events[0][0].elapsed_time(events[-1][1]),
             'decode_completion_ms':(decode_done-start)*1000,'whole_ms':(time.perf_counter()-start)*1000}
        if mode!='alone':row['competitor_span_ms']=ce[0].elapsed_time(ce[1])
        print(json.dumps(row),flush=True)
