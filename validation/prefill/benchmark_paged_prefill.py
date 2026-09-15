"""Matched old/new allocation-path benchmark; exclusive GPU test window only."""
import importlib.util
import json
import torch

spec = importlib.util.spec_from_file_location('sparse', '/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.manual_seed(17)
for batch, length, extension in [(1,32768,4096),(1,524288,8),(1,524288,4096),(4,524288,1024)]:
    h,d=2,256
    n=batch*length
    k=torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v=torch.randn(n,h,d,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    q=torch.randn(batch*extension,h*12,d,device='cuda',dtype=torch.bfloat16)
    table=torch.randperm(n,device='cuda').reshape(batch,length).int()
    req=torch.arange(batch,device='cuda',dtype=torch.int32)
    cu=torch.arange(batch+1,device='cuda',dtype=torch.int32)*extension
    cuk=torch.arange(batch+1,device='cuda',dtype=torch.int32)*length
    lens=torch.full((batch,),length,device='cuda',dtype=torch.int32)
    idx=torch.linspace(0,length-extension-1,2051,device='cuda').int().expand(batch*extension,-1).contiguous()
    def old():
        kp=[k.index_select(0,table[b].long()).to(q.dtype) for b in range(batch)]
        vp=[v.index_select(0,table[b].long()).to(q.dtype) for b in range(batch)]
        return m.sparse_gqa_fwd_interface_triton_ck(q,torch.cat(kp),torch.cat(vp),idx,cu,cuk,lens,d**-0.5)
    def new():
        return m.sparse_gqa_fwd_interface_triton_ck(q,k,v,idx,cu,None,lens,d**-0.5,req_to_token=table,req_indices=req,max_q=extension)
    a,b=old(),new()
    torch.testing.assert_close(a,b,atol=0.01,rtol=0.01)
    del a,b
    result={'batch':batch,'context_per_request':length,'extension_per_request':extension}
    for label,fn in [('old',old),('paged',new)]:
        fn();torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();out=fn();end.record();end.synchronize()
            times.append(start.elapsed_time(end));del out
        result[label]={'median_ms':sorted(times)[2],'peak_extra_MiB':(torch.cuda.max_memory_allocated()-baseline)/2**20}
    print(json.dumps(result),flush=True)
    del k,v,q,table,req,cu,cuk,lens,idx
