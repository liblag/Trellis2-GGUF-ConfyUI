import importlib.util
from pathlib import Path
import types
import unittest

spec=importlib.util.spec_from_file_location('gpu_residency',Path(__file__).resolve().parents[1]/'gpu_residency.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class ResidencyTests(unittest.TestCase):
    def pipeline(self,device):
        model=types.SimpleNamespace(named_parameters=lambda:[('weight',types.SimpleNamespace(device=device))],named_buffers=lambda:[])
        return types.SimpleNamespace(low_vram=False,device='cuda:0',models={'flow':model})

    def test_gpu_weights_pass_without_cuda_operations(self):
        mod.assert_gpu_resident(self.pipeline('cuda:0'))

    def test_cpu_weights_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError,'flow.weight is on cpu'):mod.assert_gpu_resident(self.pipeline('cpu'))

    def test_meta_weights_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError,'expected CUDA'):mod.assert_gpu_resident(self.pipeline('meta'))

    def test_explicit_offload_mode_keeps_existing_behavior(self):
        pipeline=self.pipeline('cpu');pipeline.low_vram=True
        mod.assert_gpu_resident(pipeline)

if __name__=='__main__':unittest.main()
