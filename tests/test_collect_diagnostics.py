import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('collect_diagnostics',Path(__file__).resolve().parents[1]/'colab/collect_diagnostics.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class CollectorTests(unittest.TestCase):
    def test_report_is_read_only_and_redacts_tokens(self):
        self.assertEqual(mod.redact('key=hf_example123'), 'key=[REDACTED]')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'comfy-t4.log').write_text('value=hf_example123')
            calls=[]
            def command(args,timeout=6):calls.append(args);return {'code':0,'output':'sample'}
            with patch.multiple(mod,ROOT=root,NODE=root/'node'),patch.object(mod,'command',command),\
                 patch.object(mod.metadata,'distribution',side_effect=mod.metadata.PackageNotFoundError()),\
                 patch.object(mod.urllib.request,'urlopen',side_effect=OSError('not running')):
                report=mod.collect(samples=1)
            self.assertEqual(report['comfy_log_tail'],'value=[REDACTED]')
            self.assertTrue(all(args[0] in ['nvidia-smi','git','/usr/local/cuda-12.6/bin/nvcc'] for args in calls))
            self.assertNotIn('prompt',report)

if __name__=='__main__':unittest.main()
