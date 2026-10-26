"""Update only Python node code; retain CUDA wheels, models, and a full old checkout backup."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

ROOT=Path('/content')
NODE=ROOT/'ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF'
REPO='https://github.com/liblag/Trellis2-GGUF-ConfyUI.git'

def update(ref):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8188))==0:
            raise RuntimeError('Stop the current ComfyUI process before updating node code. Do not reset the Colab runtime.')
    staging=ROOT/'t4-node-update'/uuid.uuid4().hex
    staging.mkdir(parents=True)
    def git(*args):subprocess.run(['git','-C',str(staging),*args],check=True)
    git('init');git('remote','add','origin',REPO);git('fetch','--depth','1','origin',ref);git('checkout','--detach','FETCH_HEAD')
    for name in ['nodes.py','__init__.py','execution_logging.py','gpu_residency.py',
                 'trellis2_gguf/modules/sparse/conv/explicit_gemm.py']:
        path=staging/name
        if not path.is_file():raise RuntimeError('Required updated source is missing: '+name)
        ast.parse(path.read_text(encoding='utf8'),filename=name)
    revision=subprocess.check_output(['git','-C',str(staging),'rev-parse','HEAD'],text=True).strip()
    backup=ROOT/'t4-node-backups'/f'{NODE.name}-{time.time_ns()}'
    backup.parent.mkdir(parents=True,exist_ok=True)
    NODE.parent.mkdir(parents=True,exist_ok=True)
    had_previous=NODE.exists()
    if had_previous:shutil.move(str(NODE),str(backup))
    try:shutil.move(str(staging),str(NODE))
    except BaseException:
        if had_previous and not NODE.exists():shutil.move(str(backup),str(NODE))
        raise
    for name in ['trellis-t4-bootstrap/colab/sources.lock.json','t4-installed.json','t4-verified.json']:
        path=ROOT/name
        if not path.exists():continue
        data=json.loads(path.read_text())
        sources=data.get('sources',data)
        if 'trellis' in sources:
            sources['trellis']={'url':REPO,'ref':revision}
            path.write_text(json.dumps(data,indent=2))
    print('Updated node:',revision)
    if had_previous:print('Previous checkout preserved:',backup)
    print('CUDA wheels, venv312, input and output were not changed. Start ComfyUI again.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ref',required=True)
    update(parser.parse_args().ref)
