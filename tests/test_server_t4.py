import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('server_t4',Path(__file__).resolve().parents[1]/'colab/server_t4.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class LauncherTests(unittest.TestCase):
    def test_launch_uses_wildcard_cors_and_does_not_require_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'t4-installed.json').write_text('{}')
            (root/'cloudflared-t4').touch()
            paths={'ROOT':root,'STATE':root/'state.json','COMFY':root/'ComfyUI','PY':root/'python',
                   'LOG':root/'comfy.log','TLOG':root/'tunnel.log'}
            calls=[]
            def spawn(args,log,cwd,env):
                calls.append(args)
                log.write_text('https://example-test.trycloudflare.com' if 'tunnel' in args else 'Using native ComfyUI-GGUF support')
                return types.SimpleNamespace(poll=lambda:None),{'pid':123,'identity':{}}
            def urlopen(request,timeout=None):
                info={k:{} for k in ['Trellis2LoadModel_GGUF','Trellis2PreProcessImage_GGUF','Trellis2MeshWithVoxelGenerator_GGUF']}
                result=io.BytesIO(json.dumps(info).encode());result.status=200;return result
            sock=types.SimpleNamespace(connect_ex=lambda address:1)
            socket_context=contextlib.nullcontext(sock)
            with patch.multiple(mod,**paths),patch.object(mod,'spawn',spawn),patch.object(mod,'stop',lambda:None),\
                 patch.object(mod.socket,'socket',return_value=socket_context),patch.object(mod.urllib.request,'urlopen',urlopen),\
                 contextlib.redirect_stdout(io.StringIO()) as output:
                mod.start()
            command=next(c for c in calls if 'main.py' in c)
            i=command.index('--enable-cors-header')
            self.assertEqual(command[i+1],'*')
            self.assertIn('https://example-test.trycloudflare.com',output.getvalue())
            self.assertFalse((root/'t4-verified.json').exists())

if __name__=='__main__':unittest.main()
