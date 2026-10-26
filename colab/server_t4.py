"""Start returns to the notebook; stop targets only recorded matching processes."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path('/content')
STATE=ROOT/'trellis-t4-processes.json'
COMFY=ROOT/'ComfyUI'
PY=ROOT/'venv312/bin/python'
LOG=ROOT/'comfy-t4.log'
TLOG=ROOT/'cloudflare-t4.log'

def identity(pid):
    try:
        return {'cmd':Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' '),
                'started':Path(f'/proc/{pid}/stat').read_text().split()[21]}
    except (FileNotFoundError,ProcessLookupError):return None

def stop():
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    for kind in ('tunnel','comfy'):
        record=state.get(kind)
        if not record:continue
        pid=record['pid']
        if identity(pid)!=record['identity']:
            print(kind,': recorded process already stopped or changed');continue
        os.killpg(pid,signal.SIGTERM)
        for _ in range(30):
            if identity(pid)!=record['identity']:break
            time.sleep(.1)
        if identity(pid)==record['identity']:os.killpg(pid,signal.SIGKILL)
        print('Stopped',kind)
    STATE.unlink(missing_ok=True)

def spawn(args,log,cwd,env):
    with log.open('w') as output:
        proc=subprocess.Popen(args,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    return proc,{'pid':proc.pid,'identity':identity(proc.pid)}

def start():
    if not (ROOT/'t4-verified.json').exists():raise RuntimeError('Run CUDA verification before launching.')
    stop()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8188))==0:raise RuntimeError('Port 8188 is occupied by a different server; stop its original launch cell first.')
    binary=ROOT/'cloudflared-t4'
    if not binary.exists():
        url='https://github.com/cloudflare/cloudflared/releases/download/2026.10.3/cloudflared-linux-amd64'
        tmp=binary.with_suffix('.download')
        urllib.request.urlretrieve(url,tmp);tmp.chmod(0o755);tmp.replace(binary)
    env=dict(os.environ,CUDA_HOME='/usr/local/cuda-12.6',TORCH_CUDA_ARCH_LIST='7.5',
             SPARSE_CONV_BACKEND='flex_gemm',SPARSE_ATTN_BACKEND='sdpa',ATTN_BACKEND='sdpa',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
    comfy,record=spawn([str(PY),'-u','main.py','--listen','127.0.0.1','--port','8188',
                        '--disable-pinned-memory'],LOG,COMFY,env)
    state={'comfy':record};STATE.write_text(json.dumps(state))
    try:
        for _ in range(150):
            if comfy.poll() is not None:raise RuntimeError('ComfyUI exited: '+LOG.read_text(errors='replace')[-8000:])
            try:
                with urllib.request.urlopen('http://127.0.0.1:8188/object_info',timeout=2) as response:info=json.load(response)
                required=['Trellis2LoadModel_GGUF','Trellis2PreProcessImage_GGUF','Trellis2MeshWithVoxelGenerator_GGUF']
                if any(k not in info for k in required):raise RuntimeError('Trellis nodes failed to import. See /content/comfy-t4.log')
                if 'Using native ComfyUI-GGUF support' not in LOG.read_text(errors='replace'):
                    raise RuntimeError('Native GGUF loader not active. See /content/comfy-t4.log')
                break
            except (TimeoutError,OSError):time.sleep(1)
        else:raise TimeoutError('ComfyUI did not become ready in time; see log.')
        tunnel,record=spawn([str(binary),'tunnel','--protocol','http2','--url','http://127.0.0.1:8188'],TLOG,ROOT,env)
        state['tunnel']=record;STATE.write_text(json.dumps(state))
        for _ in range(60):
            if tunnel.poll() is not None:raise RuntimeError(TLOG.read_text(errors='replace')[-4000:])
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',TLOG.read_text(errors='replace'))
            if match:
                print('\nComfyUI:',match.group(0),flush=True)
                print('Ready. This cell finishes; ComfyUI stays running. Use the stop cell before setup changes.')
                print('Templates: Workflows > Trellis2-T4. Start with Only_Mesh_Simple.')
                return
            time.sleep(1)
        raise TimeoutError('No tunnel URL; see cloudflare-t4.log')
    except BaseException:
        stop();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop'])
    action=parser.parse_args().action
    globals()[action]()
