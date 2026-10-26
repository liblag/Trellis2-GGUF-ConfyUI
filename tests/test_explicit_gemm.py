import importlib.util
import itertools
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch=None

if torch is not None:
    spec=importlib.util.spec_from_file_location('explicit_gemm',Path(__file__).resolve().parents[1]/'trellis2_gguf/modules/sparse/conv/explicit_gemm.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

@unittest.skipIf(torch is None,'PyTorch not installed in test environment')
class ExplicitGemmTests(unittest.TestCase):
    def inputs(self,dtype=None):
        dtype=dtype or torch.float32
        torch.manual_seed(4)
        coords=[p for p in itertools.product(range(3),repeat=3) if p!=(1,1,1)]
        lookup={p:i for i,p in enumerate(coords)}
        offsets=list(itertools.product(range(-1,2),repeat=3))
        neighbors=torch.tensor([[lookup.get(tuple(c[j]+delta[j] for j in range(3)),0xffffffff)
                                 for delta in offsets] for c in coords],dtype=torch.int64)
        features=torch.randn(len(coords),4,dtype=dtype)
        weights=torch.randn(5,3,3,3,4,dtype=dtype)
        bias=torch.randn(5,dtype=dtype)
        return features,neighbors,weights,bias

    def reference(self,f,n,w,b):
        # Independent per-neighbor accumulation, no im2col or chunking.
        result=f.new_zeros((len(f),w.shape[0]))
        flattened=w.reshape(w.shape[0],-1,f.shape[1])
        for row in range(len(f)):
            for kernel in range(n.shape[1]):
                index=int(n[row,kernel])
                if 0<=index<len(f):result[row]+=torch.mv(flattened[:,kernel,:],f[index])
        return result+b if b is not None else result

    def test_matches_reference_across_multiple_small_chunks(self):
        f,n,w,b=self.inputs()
        progress=[]
        with torch.no_grad():result=mod.explicit_gemm(f,n,w,b,budget_mb=.001,progress=lambda i,total:progress.append((i,total)))
        torch.testing.assert_close(result,self.reference(f,n,w,b),atol=1e-5,rtol=1e-5)
        self.assertGreater(len(progress),1)
        self.assertEqual(progress[-1][0],progress[-1][1])

    def test_uint32_padding_and_no_bias(self):
        f,n,w,_=self.inputs()
        with torch.no_grad():result=mod.explicit_gemm(f,n.to(torch.uint32),w,budget_mb=.001)
        torch.testing.assert_close(result,self.reference(f,n,w,None),atol=1e-5,rtol=1e-5)

    def test_fp16_matches_float_reference(self):
        f,n,w,b=self.inputs(torch.float16)
        with torch.no_grad():result=mod.explicit_gemm(f,n,w,b,budget_mb=.001)
        expected=self.reference(f.float(),n,w.float(),b.float()).half()
        torch.testing.assert_close(result,expected,atol=.02,rtol=.01)

    def test_training_is_not_silently_used(self):
        f,n,w,b=self.inputs();w.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError,'inference-only'):mod.explicit_gemm(f,n,w,b)

    def test_cancellation_checkpoint_is_honored(self):
        f,n,w,b=self.inputs()
        def cancel():raise InterruptedError('cancelled')
        with torch.no_grad(),self.assertRaises(InterruptedError):mod.explicit_gemm(f,n,w,b,checkpoint=cancel)

if __name__=='__main__':unittest.main()
