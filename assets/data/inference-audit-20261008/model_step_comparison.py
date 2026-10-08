"""Full trained-model fixed-step fixture; not an autoregressive serving trace.

Every decode replay reads the same real prefix KV and computes the same next
model token. This isolates full-model coexecution before serving integration.
"""
import argparse
import json
import math
import random
import time
import torch
import torch_npu
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vector_decode import VectorDecode

p=argparse.ArgumentParser()
p.add_argument('--model',default='/models/Qwen3-0.6B')
p.add_argument('--context',type=int,default=512)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--prefill-tokens',type=int,default=512)
p.add_argument('--steps',type=int,default=10)
p.add_argument('--prefill-count',type=int,default=3)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--competitor',choices=['prefill','h2d','layer_h2d','p2p','layer_p2p'],default='prefill')
p.add_argument('--kv-context',type=int,default=4096)
p.add_argument('--vendor-only',action='store_true')
p.add_argument('--device',type=int,default=0)
a=p.parse_args()
torch.manual_seed(42)
torch.npu.set_device(a.device) # default isolated container physical card 3
torch.npu.config.allow_internal_format=False
mask=torch.triu(torch.ones(2048,2048),diagonal=1).to(torch.int8).npu()
vector_kernels={}
policy='vendor'
def attention(module,q,k,v,attention_mask,dropout=0.0,scaling=None,**kwargs):
    q,k,v=(t.contiguous() for t in (q,k,v))
    b,h,qlen,d=q.shape
    kh=k.shape[1]
    seq=k.shape[2]
    if policy=='vector' and qlen==1:
        key=(module.layer_idx,b,h,seq)
        if key not in vector_kernels:vector_kernels[key]=VectorDecode(q,k,splits=4,block=32)
        out=vector_kernels[key](q,k,v)
    else:
        out=torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=h,num_key_value_heads=kh,
            input_layout='BNSD',scale=scaling,atten_mask=mask if qlen>1 else None,
            sparse_mode=3 if qlen>1 else 0,
            actual_seq_lengths=[qlen]*b,actual_seq_lengths_kv=[seq]*b)[0]
    return out.transpose(1,2).contiguous(),None
ALL_ATTENTION_FUNCTIONS.register('cjw_probe',attention)
with torch.inference_mode():
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).eval().to('npu')
    # Check the custom causal interface against HF eager on a short prompt.
    check_ids=torch.randint(1000,12000,(1,32),device='npu')
    eager=model(check_ids,use_cache=False,logits_to_keep=1).logits.float().clone()
    model.config._attn_implementation='cjw_probe'
    actual=model(check_ids,use_cache=False,logits_to_keep=1).logits.float()
    torch.npu.synchronize()
    relative_rmse=((actual-eager).square().mean()/eager.square().mean()).sqrt().item()
    greedy_equal=torch.equal(actual.argmax(-1),eager.argmax(-1))
    print(json.dumps({'model':a.model,'eager_interface_validation':{'relative_rmse':relative_rmse,'greedy_equal':greedy_equal}}),flush=True)
    if relative_rmse>.02 or not greedy_equal:raise RuntimeError('full-model attention interface validation failed')
    prefix_ids=torch.randint(1000,12000,(a.batch,a.context-1),device='npu')
    prefix_output=model(prefix_ids,use_cache=True,logits_to_keep=1)
    prefix=[(layer.keys,layer.values) for layer in prefix_output.past_key_values.layers]
    input_ids=torch.randint(1000,12000,(a.batch,1),device='npu')
    positions=torch.full((a.batch,1),a.context-1,device='npu',dtype=torch.long)
    prefill_ids=torch.randint(1000,12000,(1,a.prefill_tokens),device='npu')
    def decode():
        cache=DynamicCache(ddp_cache_data=prefix,config=model.config)
        return model(input_ids,position_ids=positions,past_key_values=cache,
                     use_cache=True,logits_to_keep=1).logits
    if a.competitor!='prefill':
        kv_shape=(model.config.num_hidden_layers,2,a.kv_context,model.config.num_key_value_heads,model.config.head_dim)
        host_kv=torch.ones(kv_shape,dtype=torch.bfloat16,pin_memory=True)
        source_kv=host_kv.to('npu:0') if 'p2p' in a.competitor else host_kv
        target_kv=torch.empty(kv_shape,dtype=torch.bfloat16,device='npu')
        print(json.dumps({'kv_transfer_bytes':host_kv.numel()*host_kv.element_size(),'pinned':host_kv.is_pinned()}),flush=True)
    def prefill():
        if a.competitor=='prefill':return model(prefill_ids,use_cache=False,logits_to_keep=1).logits
        if a.competitor in ['h2d','p2p']:target_kv.copy_(source_kv,non_blocking=True)
        else:
            for layer in range(model.config.num_hidden_layers):target_kv[layer].copy_(source_kv[layer],non_blocking=True)
        return target_kv
    reference=decode().float().clone()
    configs={}
    for name,cube in [('vendor',None),('reserve4',4),('vector',None)]:
        if a.vendor_only and name!='vendor':continue
        policy='vector' if name=='vector' else 'vendor'
        ds=torch.npu.Stream();cs=torch.npu.Stream()
        if cube:
            torch.npu.set_stream_limit(ds,cube_num=cube,vector_num=2*cube)
            torch.npu.set_stream_limit(cs,cube_num=20-cube,vector_num=2*(20-cube))
        with torch.npu.stream(ds):
            for _ in range(2):decode()
        with torch.npu.stream(cs):prefill()
        torch.npu.synchronize()
        dg=torch.npu.NPUGraph();pg=torch.npu.NPUGraph()
        with torch.npu.graph(dg,stream=ds):dout=decode()
        if a.competitor=='prefill':
            with torch.npu.graph(pg,stream=cs):pout=prefill()
        else:
            pg=None;pout=target_kv
        dg.replay();torch.npu.synchronize()
        error=(dout.float()-reference).abs().max().item()
        rms=((dout.float()-reference).square().mean()/reference.square().mean()).sqrt().item()
        equal=torch.equal(dout.argmax(-1),reference.argmax(-1))
        print(json.dumps({'config':name,'max_logit_abs_error':error,'relative_rmse':rms,'greedy_equal':equal}),flush=True)
        if rms>.02 or not equal:
            print(json.dumps({'config':name,'excluded':'decode graph numerical failure'}),flush=True)
            continue
        configs[name]=(ds,cs,dg,pg,dout,pout)
    rng=random.Random(2026)
    for repeat in range(a.repeats):
        cases=[(name,mode) for name in configs for mode in ['alone','serial','competitor_first','decode_first']]
        rng.shuffle(cases)
        for name,mode in cases:
            ds,cs,dg,pg,_,_=configs[name]
            torch.npu.synchronize()
            events=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(a.steps)]
            cbegin=torch.npu.Event(enable_timing=True);cend=torch.npu.Event(enable_timing=True)
            start=time.perf_counter()
            def compete():
                with torch.npu.stream(cs):
                    cbegin.record()
                    for _ in range(a.prefill_count):
                        if pg is not None:pg.replay()
                        else:prefill()
                    cend.record()
            if mode in ['serial','competitor_first']:compete()
            if mode=='serial':cs.synchronize()
            with torch.npu.stream(ds):
                for begin,end in events:begin.record();dg.replay();end.record()
            if mode=='decode_first':compete()
            events[-1][1].synchronize()
            done=time.perf_counter()
            torch.npu.synchronize()
            whole_ms=(time.perf_counter()-start)*1000
            if mode!='alone' and a.competitor!='prefill' and not torch.equal(target_kv[:, :, -1, 0, :].cpu(),host_kv[:, :, -1, 0, :]):raise RuntimeError('KV copy numerical check failed')
            print(json.dumps({'config':name,'mode':mode,'repeat':repeat,'metadata':vars(a),
                'competitor_ms':None if mode=='alone' else cbegin.elapsed_time(cend),
                'op_ms':[b.elapsed_time(e) for b,e in events],
                'decode_span_ms':events[0][0].elapsed_time(events[-1][1]),
                'decode_completion_ms':(done-start)*1000,'whole_ms':whole_ms}),flush=True)
