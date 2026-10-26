"""Optional persistent cache of compiled wheels, keyed by source and runtime ABI."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import uuid


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


class WheelCache:
    def __init__(self, root=None):
        self.root=Path(root) if root else None

    def directory(self, package, identity):
        if not self.root:return None
        runtime={'machine':platform.machine(),'libc':platform.libc_ver()}
        key=hashlib.sha256(json.dumps({'identity':identity,'runtime':runtime},sort_keys=True).encode()).hexdigest()[:24]
        return self.root/package/key

    def restore(self, package, identity, destination):
        directory=self.directory(package,identity)
        if directory is None or not (directory/'manifest.json').exists():return None
        try:
            info=json.loads((directory/'manifest.json').read_text())
        except (OSError,ValueError):
            print('Unreadable persistent cache manifest:',directory,flush=True);return None
        if info.get('identity')!=identity:return None
        name=info.get('wheel','')
        if Path(name).name!=name or not name.endswith('.whl'):raise ValueError('Invalid wheel cache filename')
        source=directory/name
        if not source.exists() or sha256(source)!=info['sha256']:
            print('Persistent wheel cache is incomplete/corrupt:',source,flush=True);return None
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        target=destination/name
        shutil.copy2(source,target)
        print('Restored compiled wheel:',target.name,flush=True)
        return target

    def save(self, package, identity, wheel):
        directory=self.directory(package,identity)
        if directory is None:return
        directory.mkdir(parents=True,exist_ok=True)
        wheel=Path(wheel);target=directory/wheel.name
        digest=sha256(wheel)
        if not target.exists() or sha256(target)!=digest:
            temporary=directory/(wheel.name+'.'+uuid.uuid4().hex+'.tmp')
            shutil.copy2(wheel,temporary);temporary.replace(target)
        manifest={'identity':identity,'wheel':wheel.name,'sha256':digest}
        temp=directory/('manifest.'+uuid.uuid4().hex+'.tmp')
        temp.write_text(json.dumps(manifest,indent=2));temp.replace(directory/'manifest.json')
        print('Persistent wheel saved:',target,flush=True)
