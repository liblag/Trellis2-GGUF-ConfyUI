"""Actual GPU operations, not just extension imports. No model download."""
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
import flex_gemm
import cumesh
import o_voxel
import nvdiffrast.torch as dr
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, set_algorithm

assert torch.__version__=='2.9.1+cu126',torch.__version__
assert torch.cuda.get_device_capability()==(7,5),'This notebook targets T4'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
coords=torch.tensor([[0,x,y,z] for x in range(4) for y in range(4) for z in range(4)],device='cuda',dtype=torch.int32)
features=torch.randn(len(coords),32,device='cuda',dtype=torch.float16)
weights=torch.randn(32,3,3,3,32,device='cuda',dtype=torch.float16)*0.01
bias=torch.zeros(32,device='cuda',dtype=torch.float16)
for algo in ['explicit_gemm']:
    set_algorithm(algo)
    with torch.no_grad():out,_=sparse_submanifold_conv3d(features,coords,torch.Size([1,32,4,4,4]),weights,bias)
    torch.cuda.synchronize()
    assert out.shape==(64,32) and torch.isfinite(out).all()
    print('PASS flex_gemm:',algo,flush=True)
import importlib.util
from pathlib import Path
source=Path('/content/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/modules/sparse/conv/explicit_gemm.py')
if source.exists():
    from flex_gemm.ops.spconv import SubMConv3dFunction
    spec=importlib.util.spec_from_file_location('t4_explicit_gemm',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache=SubMConv3dFunction._compute_neighbor_cache(coords,torch.Size([1,32,4,4,4]),(3,3,3),(1,1,1),False)
    with torch.no_grad():chunked=module.explicit_gemm(features,cache['neighbor_map'],weights,bias,budget_mb=.01)
    torch.cuda.synchronize()
    torch.testing.assert_close(chunked,out,atol=.01,rtol=.01)
    assert chunked.device.type=='cuda'
    print('PASS bounded explicit sparse GEMM on CUDA, matches upstream explicit result',flush=True)
vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],device='cuda')
faces=torch.tensor([[0,1,2]],device='cuda',dtype=torch.int32)
mesh=cumesh.CuMesh();mesh.init(vertices,faces);v,f=mesh.read();torch.cuda.synchronize()
assert v.shape[0]==3 and f.shape[0]==1
print('PASS CuMesh init/read',flush=True)
context=dr.RasterizeCudaContext()
clip=torch.tensor([[[-.5,-.5,0.,1.],[.5,-.5,0.,1.],[0.,.5,0.,1.]]],device='cuda')
rast,_=dr.rasterize(context,clip,faces,resolution=[16,16]);torch.cuda.synchronize()
assert (rast[...,3]>0).any()
print('PASS nvdiffrast CUDA rasterization; o_voxel import passed (export operation not tested)',flush=True)
