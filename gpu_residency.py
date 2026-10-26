"""Validate registered weights/buffers without moving tensors or synchronizing CUDA."""


def assert_gpu_resident(pipeline):
    if getattr(pipeline,'low_vram',True) or not str(getattr(pipeline,'device','')).startswith('cuda'):
        return
    models=dict(getattr(pipeline,'models',{}))
    for name in ('image_cond_model','rembg_model'):
        model=getattr(pipeline,name,None)
        if model is not None:models[name]=getattr(model,'model',model)
    checked=0
    for name,model in models.items():
        if model is None:continue
        if not hasattr(model,'named_parameters'):
            model=getattr(model,'model',model)
        if not hasattr(model,'named_parameters'):continue
        for category in ('named_parameters','named_buffers'):
            method=getattr(model,category,None)
            if method is None:continue
            for tensor_name,tensor in method():
                if tensor is None:continue
                device=str(tensor.device)
                if not device.startswith('cuda'):
                    raise RuntimeError(f'GPU-resident mode: {name}.{tensor_name} is on {device}, expected CUDA. CPU offload is disabled.')
                checked+=1
    print(f'[Trellis] GPU residency verified: {checked} registered parameter/buffer tensors on CUDA; CPU offload disabled.',flush=True)
