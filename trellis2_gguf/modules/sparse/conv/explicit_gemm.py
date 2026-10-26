"""Bounded-workspace explicit sparse GEMM. Input, gather and matmul stay on their device."""
import math
import torch
import torch.nn.functional as F


def chunk_rows(channels, kernel_volume, output_channels, element_size, budget_mb=128):
    # Account for im2col, gathered features, indices/mask and per-chunk output.
    per_row=(2*kernel_volume*channels+output_channels)*element_size+kernel_volume*17
    return max(1,min(65536,int(budget_mb*1024**2)//max(1,per_row)))


def explicit_gemm(features, neighbor_map, weight, bias=None, budget_mb=128, checkpoint=None, progress=None):
    """Same im2col + linear operation as upstream EXPLICIT_GEMM, in bounded row chunks.

    Inference-only: training/gradient callers must use the original autograd path.
    Neighbor map contains global feature indices; 0xffffffff/-1 entries mean padding.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (features,weight,bias) if t is not None):
        raise RuntimeError('Chunked explicit GEMM is inference-only')
    if features.device!=weight.device or features.device!=neighbor_map.device:
        raise ValueError('Features, neighbor map and weights must share a device')
    n,ci=features.shape
    co=weight.shape[0]
    volume=math.prod(weight.shape[1:-1])
    if weight.shape[-1]!=ci or neighbor_map.numel()!=n*volume:
        raise ValueError('Sparse convolution shape mismatch')
    rows=chunk_rows(ci,volume,co,features.element_size(),budget_mb)
    indices=neighbor_map.reshape(n,volume)
    matrix=weight.reshape(co,volume*ci)
    result=features.new_empty((n,co))
    total=math.ceil(n/rows)
    for part,start in enumerate(range(0,n,rows),1):
        if checkpoint:checkpoint()
        end=min(start+rows,n)
        current=indices[start:end].reshape(-1).long()
        valid=(current>=0)&(current<n)
        expanded=features.new_zeros(((end-start)*volume,ci))
        expanded[valid]=features[current[valid]]
        result[start:end]=F.linear(expanded.reshape(end-start,volume*ci),matrix,bias)
        if progress:progress(part,total)
        # Do not retain the previous chunk while allocating the next one.
        del expanded,current,valid
    return result
