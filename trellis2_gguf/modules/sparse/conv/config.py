import os

SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
# auto selects bounded explicit GEMM on pre-Ampere GPUs, including T4 (sm_75).
FLEX_GEMM_ALGO = os.environ.get('TRELLIS_FLEX_GEMM_ALGO','auto')
FLEX_GEMM_WORKSPACE_MB = max(8,int(os.environ.get('TRELLIS_FLEX_WORKSPACE_MB','128')))
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
