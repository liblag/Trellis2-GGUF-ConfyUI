"""Staged setup for the accompanying Colab notebook. Never changes system Python."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/content')
ASSETS = Path(__file__).resolve().parent
PY = ROOT/'venv312/bin/python'
COMFY = ROOT/'ComfyUI'
NODE = COMFY/'custom_nodes/ComfyUI-Trellis2-GGUF'
CONSTRAINTS = ROOT/'torch_constraints.txt'
LOCK = json.loads((ASSETS/'sources.lock.json').read_text())
ENV = dict(os.environ, CUDA_HOME='/usr/local/cuda-12.6', TORCH_CUDA_ARCH_LIST='7.5', MAX_JOBS='1',
           SPARSE_CONV_BACKEND='flex_gemm', SPARSE_ATTN_BACKEND='sdpa', ATTN_BACKEND='sdpa',
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
ENV['PATH'] = ENV['CUDA_HOME']+'/bin:'+ENV.get('PATH','')
ENV.update(PYTHONUNBUFFERED='1', PIP_PROGRESS_BAR='off', NINJA_STATUS='[%f/%t %es] ')

def build_resources():
    data={'disk_free_gb':round(shutil.disk_usage(ROOT).free/1024**3,2)}
    for filename in ('memory.events','memory.current','memory.max'):
        path=Path('/sys/fs/cgroup')/filename
        if path.exists():data[filename]=path.read_text().strip()
    meminfo=Path('/proc/meminfo')
    if meminfo.exists():
        data['system_memory']='; '.join(line for line in meminfo.read_text().splitlines()
                                      if line.startswith(('MemAvailable:','SwapFree:')))
    print('BUILD RESOURCES:',json.dumps(data),flush=True)
    return data

def run(args, cwd=None, capture=False):
    print('+', ' '.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), cwd=cwd, env=ENV, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout

def pip(*args):
    run([PY,'-m','pip',*args,'-c',CONSTRAINTS])

def checkout(key, path):
    source=LOCK[key]
    path.parent.mkdir(parents=True,exist_ok=True)
    if key=='trellis' and (path/'.git').exists():
        origin=run(['git','remote','get-url','origin'],cwd=path,capture=True).strip().removesuffix('.git').lower()
        wanted=source['url'].removesuffix('.git').lower()
        if origin!=wanted:
            # Keep the whole previous node, its Git metadata and local changes outside custom_nodes.
            backup=ROOT/'t4-node-backups'/f'{path.name}-{time.time_ns()}'
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),str(backup))
            print('Previous Trellis node preserved at',backup,flush=True)
    if not (path/'.git').exists():
        if path.exists() and any(path.iterdir()):raise RuntimeError(f'{path} is not a Git checkout')
        run(['git','clone','--filter=blob:none',source['url'],path])
    current=run(['git','rev-parse','HEAD'],cwd=path,capture=True).strip() if (path/'.git/HEAD').exists() else ''
    # Never reset an existing checkout containing local work.
    dirty=run(['git','status','--porcelain'],cwd=path,capture=True)
    if current != source['ref']:
        if dirty and current:raise RuntimeError(f'{path} has local changes and differs from locked revision. Preserve them before switching.')
        run(['git','fetch','origin',source['ref']],cwd=path)
        run(['git','checkout','--detach',source['ref']],cwd=path)
    run(['git','submodule','update','--init','--recursive'],cwd=path)
    return path

def system():
    run(['nvidia-smi'])
    run(['apt-get','update','-qq'])
    run(['apt-get','install','-y','software-properties-common','build-essential','git','pkg-config',
         'libgl1','libegl1','libglib2.0-0','libxrender1','libxi6','libgomp1'])
    if not shutil.which('python3.12'):
        run(['add-apt-repository','-y','ppa:deadsnakes/ppa'])
        run(['apt-get','update','-qq'])
    run(['apt-get','install','-y','python3.12','python3.12-venv','python3.12-dev'])
    if not Path('/usr/local/cuda-12.6/bin/nvcc').exists():
        # Use NVIDIA's already configured Colab package repository; fail clearly if absent.
        run(['apt-get','install','-y','cuda-toolkit-12-6'])
    run(['/usr/local/cuda-12.6/bin/nvcc','--version'])
    if not PY.exists():run(['python3.12','-m','venv',PY.parent.parent])
    run([PY,'-m','pip','install','--upgrade','pip','setuptools','wheel','ninja'])
    CONSTRAINTS.write_text('torch==2.9.1+cu126\ntorchvision==0.24.1+cu126\ntorchaudio==2.9.1+cu126\n')
    pip('install','torch==2.9.1+cu126','torchvision==0.24.1+cu126','torchaudio==2.9.1+cu126',
        '--index-url','https://download.pytorch.org/whl/cu126')
    run([PY,'-c',"import torch,torchvision,torchaudio; assert torch.cuda.get_device_capability()==(7,5), 'Select T4 runtime'; print(torch.__version__,torch.cuda.get_device_name(0)); print((torch.ones(4,device='cuda')+1).cpu())"])

def dependencies():
    # Only network checkouts run in parallel; package installs and builds stay ordered.
    paths={'comfy':COMFY,'flex':ROOT/'t4-sources/flex','cumesh':ROOT/'t4-sources/cumesh',
           'voxel':ROOT/'t4-sources/voxel','diffrast':ROOT/'t4-sources/diffrast',
           'utils3d':ROOT/'t4-sources/utils3d'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item:checkout(*item),paths.items()))
    checkout('trellis',NODE)
    gguf=checkout('gguf',COMFY/'custom_nodes/ComfyUI-GGUF')
    pip('install','-r',COMFY/'requirements.txt')
    pip('install','-r',gguf/'requirements.txt')
    # The Git utils3d dependency is installed from its pinned revision below.
    lines=(NODE/'requirements.txt').read_text().splitlines()
    req=ROOT/'trellis_t4_requirements.txt'
    req.write_text('\n'.join(x for x in lines if x.strip() and not x.strip().startswith('git+'))+'\n')
    pip('install','-r',req,'rembg[cpu]','easydict','plyfile','zstandard','ninja','packaging')
    utils=checkout('utils3d',ROOT/'t4-sources/utils3d')
    pip('install',utils)
    run([PY,ASSETS.parent/'install_patch.py',NODE])
    target=COMFY/'user/default/workflows/Trellis2-T4'
    target.mkdir(parents=True,exist_ok=True)
    for f in (ASSETS.parent/'example_workflows').glob('*.json'):shutil.copy2(f,target/f.name)

def builds():
    if not PY.exists():raise RuntimeError('Run system setup first')
    wheelhouse=ROOT/'t4-wheelhouse';wheelhouse.mkdir(exist_ok=True)
    for key,subdir,distribution in [('flex','', 'flex_gemm'),('cumesh','','cumesh'),
                                    ('voxel','o-voxel','o_voxel'),('diffrast','','nvdiffrast')]:
        source=checkout(key,ROOT/'t4-sources'/key)/subdir
        marker=wheelhouse/(key+'-build.json')
        identity={'ref':LOCK[key]['ref'],'torch':'2.9.1+cu126','python':'3.12','arch':'7.5','cuda':'12.6'}
        info=json.loads(marker.read_text()) if marker.exists() else {}
        wheel=Path(info.get('wheel','/not-built'))
        if info.get('identity')!=identity or not wheel.exists():
            out=wheelhouse/key;out.mkdir(exist_ok=True)
            print(f'BUILD START: {key} | source={source} | MAX_JOBS=1 | arch=7.5',flush=True)
            build_resources()
            try:
                # Verbose PEP517 output is essential: quiet pip can report "No available output".
                # Keep build directories so ninja can reuse successfully compiled objects.
                pip('wheel','--verbose','--no-clean',source,'--no-build-isolation','--no-deps','--wheel-dir',out)
            finally:
                print(f'BUILD END/FAIL: {key}',flush=True)
                build_resources()
            candidates=list(out.glob('*.whl'))
            if len(candidates)!=1:raise RuntimeError(f'Expected one wheel in {out}; found {len(candidates)}')
            wheel=candidates[0]
            marker.write_text(json.dumps({'identity':identity,'wheel':str(wheel)},indent=2))
        # No prebuilt Torch291 wheel and no transitive Git CUDA rebuilds.
        pip('install','--force-reinstall','--no-deps',wheel)
    # Optional PBR preview renderer only: do not install an unchecked architecture-specific wheel.
    print('CUDA packages built for sm_75. Optional nvdiffrec PBR preview is not installed.',flush=True)

def verify():
    run([PY,ASSETS/'smoke_t4.py'])
    freeze=run([PY,'-m','pip','freeze'],capture=True)
    (ROOT/'t4-environment.txt').write_text(freeze)
    manifest={'sources':LOCK,'torch':'2.9.1+cu126','gpu_arch':'7.5'}
    (ROOT/'t4-verified.json').write_text(json.dumps(manifest,indent=2))
    print('CUDA smoke tests passed. Full TRELLIS generation is still required to validate end-to-end output.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['system','dependencies','builds','verify'])
    args=ap.parse_args();globals()[args.stage]()
