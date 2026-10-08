"""Experimental exact FP16 decode attention using vector reductions only.

No dot instruction; profiler must still verify the generated kernel type.
This prototype uses contiguous KV and has no production paged-cache support.
"""
import math
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def partial_attention(Q,K,V,P,L,M,SEQ:tl.constexpr,H:tl.constexpr,KH:tl.constexpr,
                      SPLITS:tl.constexpr,CHUNK:tl.constexpr,BLOCK:tl.constexpr):
    bh=tl.program_id(0)
    split=tl.program_id(1)
    batch=bh//H
    head=bh%H
    kvhead=head//(H//KH)
    d=tl.arange(0,128)
    r=tl.arange(0,BLOCK)
    q=tl.load(Q+bh*128+d).to(tl.float32)
    m=tl.full((),-float('inf'),tl.float32)
    l=tl.full((),0,tl.float32)
    acc=tl.full((128,),0,tl.float32)
    for start in range(0,CHUNK,BLOCK):
        pos=split*CHUNK+start+r
        offset=((batch*KH+kvhead)*SEQ+pos[:,None])*128+d[None,:]
        k=tl.load(K+offset,mask=pos[:,None]<SEQ,other=0).to(tl.float32)
        scores=tl.sum(k*q[None,:],1)*0.08838834764831845
        scores=tl.where(pos<SEQ,scores,-float('inf'))
        newm=tl.maximum(m,tl.max(scores,0))
        correction=tl.exp(m-newm)
        p=tl.exp(scores-newm)
        vv=tl.load(V+offset,mask=pos[:,None]<SEQ,other=0).to(tl.float32)
        acc=acc*correction+tl.sum(p[:,None]*vv,0)
        l=l*correction+tl.sum(p,0)
        m=newm
    tl.store(P+(bh*SPLITS+split)*128+d,acc)
    tl.store(L+bh*SPLITS+split,l)
    tl.store(M+bh*SPLITS+split,m)

@triton.jit
def grouped_partial_attention(Q,K,V,P,L,M,SEQ:tl.constexpr,H:tl.constexpr,KH:tl.constexpr,
                              SPLITS:tl.constexpr,CHUNK:tl.constexpr,BLOCK:tl.constexpr,G:tl.constexpr):
    bkh=tl.program_id(0)
    split=tl.program_id(1)
    batch=bkh//KH
    kvhead=bkh%KH
    g=tl.arange(0,G)
    d=tl.arange(0,128)
    r=tl.arange(0,BLOCK)
    bh=batch*H+kvhead*G+g
    q=tl.load(Q+bh[:,None]*128+d[None,:]).to(tl.float32)
    m=tl.full((G,),-float('inf'),tl.float32)
    l=tl.full((G,),0,tl.float32)
    acc=tl.full((G,128),0,tl.float32)
    for start in range(0,CHUNK,BLOCK):
        pos=split*CHUNK+start+r
        offset=((batch*KH+kvhead)*SEQ+pos[:,None])*128+d[None,:]
        k=tl.load(K+offset,mask=pos[:,None]<SEQ,other=0).to(tl.float32)
        scores=tl.sum(k[None,:,:]*q[:,None,:],2)*0.08838834764831845
        scores=tl.where(pos[None,:]<SEQ,scores,-float('inf'))
        newm=tl.maximum(m,tl.max(scores,1))
        correction=tl.exp(m-newm)
        p=tl.exp(scores-newm[:,None])
        vv=tl.load(V+offset,mask=pos[:,None]<SEQ,other=0).to(tl.float32)
        acc=acc*correction[:,None]+tl.sum(p[:,:,None]*vv[None,:,:],1)
        l=l*correction+tl.sum(p,1)
        m=newm
    tl.store(P+(bh[:,None]*SPLITS+split)*128+d[None,:],acc)
    tl.store(L+bh*SPLITS+split,l)
    tl.store(M+bh*SPLITS+split,m)

@triton.jit
def merge_attention(P,L,M,O,SPLITS:tl.constexpr):
    bh=tl.program_id(0)
    s=tl.arange(0,SPLITS)
    d=tl.arange(0,128)
    m=tl.load(M+bh*SPLITS+s)
    correction=tl.exp(m-tl.max(m,0))
    l=tl.load(L+bh*SPLITS+s)
    values=tl.load(P+(bh*SPLITS+s[:,None])*128+d[None,:])
    out=tl.sum(values*correction[:,None],0)/tl.sum(l*correction,0)
    tl.store(O+bh*128+d,out)

class VectorDecode:
    def __init__(self,q,k,splits=4,block=32,grouped=False):
        b,h,_,d=q.shape
        self.grouped=grouped
        self.seq=k.shape[2]
        self.h=h
        self.kh=k.shape[1]
        self.splits=splits
        self.block=block
        self.chunk=triton.cdiv(self.seq,splits)
        if self.chunk%block:
            raise ValueError('chunk must be aligned for this prototype')
        self.partial=torch.empty((b*h,splits,128),device=q.device,dtype=torch.float32)
        self.l=torch.empty((b*h,splits),device=q.device,dtype=torch.float32)
        self.m=torch.empty_like(self.l)
        self.out=torch.empty_like(q)
    def __call__(self,q,k,v):
        if self.grouped:
            grouped_partial_attention[(q.shape[0]*self.kh,self.splits)](q,k,v,self.partial,self.l,self.m,
                self.seq,self.h,self.kh,self.splits,self.chunk,self.block,self.h//self.kh)
        else:
            partial_attention[(q.shape[0]*self.h,self.splits)](q,k,v,self.partial,self.l,self.m,
                self.seq,self.h,self.kh,self.splits,self.chunk,self.block)
        merge_attention[(q.shape[0]*self.h,)](self.partial,self.l,self.m,self.out,self.splits)
        return self.out

if __name__=='__main__':
    import json
    torch.npu.set_device(0) # container logical 0 maps to authorized physical card 3
    q=torch.randn(1,32,1,128,device='npu',dtype=torch.float16)
    k=torch.randn(1,8,512,128,device='npu',dtype=torch.float16)
    v=torch.randn_like(k)
    kernel=VectorDecode(q,k)
    out=kernel(q,k,v)
    ref=torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=32,num_key_value_heads=8,
        input_layout='BNSD',scale=1/math.sqrt(128),actual_seq_lengths_kv=[512])[0]
    torch.npu.synchronize()
    print(json.dumps({'max_error':(out-ref).abs().max().item(),'allclose':torch.allclose(out,ref,atol=.003,rtol=.03)}),flush=True)
