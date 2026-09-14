"""Check compiled dependency ordering and routing against a settled reference."""
import importlib,json,os,re
from pathlib import Path
import torch
m=importlib.import_module("sglang.kernels.ops.moe.moe_fused_gate")
compiled=[]
original=m._router_triton_kernel.run
def capture(*args,**kwargs):
    result=original(*args,**kwargs)
    if result is not None:compiled.append(result)
    return result
m._router_triton_kernel.run=capture
for scoring in ("softmax","sigmoid"):
    torch.manual_seed(321)
    scores=torch.randn((32,512),device="cuda",dtype=torch.float32)
    bias=torch.randn(512,device="cuda",dtype=torch.float32)*0.01
    weights,ids=m.moe_fused_gate(scores,bias,10,scoring_func=scoring,num_expert_group=2,topk_group=2)
    activated=torch.softmax(scores+bias,dim=-1) if scoring=="softmax" else torch.sigmoid(scores)
    ranked=activated if scoring=="softmax" else activated+bias
    expected_ids=torch.topk(ranked,10,dim=-1).indices
    expected_weights=activated.gather(1,expected_ids)
    expected_weights=expected_weights/expected_weights.sum(dim=-1,keepdim=True)
    torch.testing.assert_close(ids.long(),expected_ids)
    torch.testing.assert_close(weights,expected_weights,atol=2e-6,rtol=2e-5)
    ptx=compiled[-1].asm["ptx"]
    lines=ptx.splitlines()
    waits=[i for i,line in enumerate(lines) if "griddepcontrol.wait" in line]
    loads=[i for i,line in enumerate(lines) if re.search(r"\bld\.global",line)]
    report={"scoring":scoring,"numerics_pass":True,"first_wait":waits[0] if waits else None,"first_global_load":loads[0] if loads else None,"ordering_pass":bool(waits and loads and min(loads)>min(waits))}
    print(json.dumps(report),flush=True)
    assert report["ordering_pass"], report
    output=Path("/out")
    if output.exists():(output/(scoring+".ptx")).write_text(ptx)

# The radix path returns winners in expert-id order.
torch.manual_seed(322)
scores = torch.randn((32, 896), device="cuda", dtype=torch.float32)
bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
assert m.moe_route_radix.covered(scores, bias, 16)
weights, ids = m.moe_fused_gate(scores, bias, 16, scoring_func="sigmoid")
activated = torch.sigmoid(scores)
expected_ids = torch.topk(activated + bias, 16, dim=-1).indices
torch.testing.assert_close(ids.long().sort(dim=-1).values, expected_ids.sort(dim=-1).values)
expected_weights = activated.gather(1, ids.long())
expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
torch.testing.assert_close(weights, expected_weights, atol=2e-6, rtol=2e-5)
print(json.dumps({"path": "radix", "numerics_pass": True}), flush=True)
