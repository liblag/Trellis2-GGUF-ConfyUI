"""Read-only report: no installs, no CUDA kernels, no model loads, no restarts."""
import argparse
import ast
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT=Path('/content')
NODE=ROOT/'ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF'

def command(args,timeout=6):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
        return {'code':p.returncode,'output':p.stdout[-16000:]}
    except (OSError,subprocess.TimeoutExpired) as error:return {'error':str(error)}

def redact(text):
    return re.sub(r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]+','[REDACTED]',text)

def text(path,limit=12000):
    try:return redact(Path(path).read_text(errors='replace')[-limit:])
    except OSError:return None

def collect(samples=10):
    report={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'note':'No CUDA inference or installations were performed.'}
    report['gpu']=command(['nvidia-smi'])
    report['nvcc']=command(['/usr/local/cuda-12.6/bin/nvcc','--version'])
    report['memory']=text('/proc/meminfo',4000)
    report['cpu']={'logical_count':os.cpu_count(),'quota':text('/sys/fs/cgroup/cpu.max'),
                   'cpuset':text('/sys/fs/cgroup/cpuset.cpus.effective')}
    cpuinfo=text('/proc/cpuinfo',100000) or ''
    report['cpu']['model']=next((line.split(':',1)[1].strip() for line in cpuinfo.splitlines()
                               if line.startswith('model name')),None)
    process_pid=None
    try:
        state=json.loads((ROOT/'trellis-t4-processes.json').read_text())
        process_pid=int(state['comfy']['pid'])
        report['comfy_pid']=process_pid
    except (OSError,ValueError,KeyError):pass
    report['cgroup']={name:text('/sys/fs/cgroup/'+name) for name in ['memory.events','memory.current','memory.max']}
    report['packages']={}
    for name in ['torch','torchvision','torchaudio','triton','flex_gemm','cumesh','o_voxel','nvdiffrast','gguf','rembg','onnxruntime','onnxruntime-gpu','comfyui-frontend-package']:
        try:
            package=metadata.distribution(name)
            direct=json.loads(package.read_text('direct_url.json') or '{}')
            archive=direct.get('archive_info',{})
            report['packages'][name]={'version':package.version,'sha256':archive.get('hashes',{}).get('sha256')}
            if name=='torch':
                path=Path(package.locate_file('torch/version.py'))
                # Read static CUDA build version without importing torch/initializing CUDA.
                tree=ast.parse(path.read_text())
                for item in tree.body:
                    if isinstance(item,ast.AnnAssign) and isinstance(item.target,ast.Name) and item.target.id=='cuda':
                        report['torch_cuda_build']=ast.literal_eval(item.value)
        except metadata.PackageNotFoundError:report['packages'][name]=None
        except Exception as error:report['packages'][name]={'error':str(error)}
    report['node_commit']=command(['git','-C',str(NODE),'rev-parse','HEAD'])
    report['node_config']={name:text(NODE/name,5000) for name in [
        'trellis2_gguf/modules/sparse/conv/config.py','trellis2_gguf/modules/sparse/config.py']}
    try:
        with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=3) as response:queue=json.load(response)
        report['queue']={'running':len(queue.get('queue_running',[])),'pending':len(queue.get('queue_pending',[]))}
        settings={}
        allowed={'device','backend','model_format','modelname','low_vram','keep_models_loaded','pipeline_type',
                 'max_num_tokens','sparse_structure_resolution','use_tiled_decoder','generate_texture_slat','shape_steps','sparse_structure_steps'}
        for item in queue.get('queue_running',[]):
            prompt=item[2]
            for node_id,node in prompt.items():
                if node.get('class_type','').startswith('Trellis'):
                    settings[node_id]={'type':node['class_type'],'settings':{k:v for k,v in node.get('inputs',{}).items() if k in allowed}}
        report['running_settings']=settings
    except Exception as error:report['queue_error']=str(error)
    report['samples']=[]
    for index in range(samples):
        row={'index':index,'gpu':command(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,pstate','--format=csv,noheader,nounits']),
             'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'])}
        if process_pid:
            status=text(f'/proc/{process_pid}/status',10000) or ''
            row['comfy_process']=[line for line in status.splitlines() if line.startswith(('State:','VmRSS:','VmPeak:','Threads:'))]
        report['samples'].append(row)
        print(f'GPU sample {index+1}/{samples}: '+row['gpu'].get('output',str(row['gpu'])).strip(),flush=True)
        if index>=1 and all('error' in r['gpu'] for r in report['samples'][-2:]):
            report['sampling_note']='Stopped after two consecutive nvidia-smi timeouts/errors.'
            break
        if index+1<samples:time.sleep(1)
    report['comfy_log_tail']=text(ROOT/'comfy-t4.log',22000)
    report['installed_manifest']=text(ROOT/'t4-installed.json',5000)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=10)
    args=parser.parse_args()
    report=collect(max(1,min(args.samples,30)))
    target=ROOT/'trellis-diagnostics.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('Report saved:',target,flush=True)
