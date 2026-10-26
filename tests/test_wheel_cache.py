import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('wheel_cache',Path(__file__).resolve().parents[1]/'colab/wheel_cache.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class WheelCacheTests(unittest.TestCase):
    def test_restore_and_source_revision_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wheel=root/'package.whl';wheel.write_bytes(b'compiled artifact')
            identity={'ref':'abc','torch':'2.9.1+cu126','arch':'7.5'}
            cache=mod.WheelCache(root/'persistent');cache.save('cumesh',identity,wheel)
            restored=cache.restore('cumesh',identity,root/'new-session')
            self.assertEqual(restored.read_bytes(),wheel.read_bytes())
            self.assertIsNone(cache.restore('cumesh',{**identity,'ref':'different'},root/'other'))

    def test_corrupt_cache_is_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wheel=root/'package.whl';wheel.write_bytes(b'good')
            cache=mod.WheelCache(root/'persistent');identity={'ref':'abc'}
            cache.save('cumesh',identity,wheel)
            (cache.directory('cumesh',identity)/wheel.name).write_bytes(b'corrupt')
            self.assertIsNone(cache.restore('cumesh',identity,root/'new-session'))

if __name__=='__main__':unittest.main()
