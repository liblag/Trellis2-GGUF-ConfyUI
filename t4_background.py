"""Cancellable rembg worker. Run with the same Python environment as ComfyUI."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def remove_background(image, timeout=180):
    from PIL import Image
    import numpy as np
    import comfy.model_management as mm

    image = image.convert('RGBA')
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    if np.any(np.asarray(image.getchannel('A')) < 255):
        print('[T4 rembg] Existing alpha detected: keeping supplied cutout.', flush=True)
        return image
    with tempfile.TemporaryDirectory(prefix='trellis-rembg-') as tmp:
        src, dst = Path(tmp)/'input.png', Path(tmp)/'output.png'
        image.save(src)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
        print(f'[T4 rembg] CPU worker starting; timeout {timeout}s. First run may download u2net.', flush=True)
        proc = subprocess.Popen([sys.executable, '-u', __file__, str(src), str(dst)], env=env)
        started = time.monotonic()
        try:
            while proc.poll() is None:
                mm.throw_exception_if_processing_interrupted()
                if time.monotonic()-started > timeout:
                    raise TimeoutError(f'rembg exceeded {timeout}s. Check download/network messages above; retry after the model has downloaded.')
                time.sleep(0.2)
            if proc.returncode:
                raise RuntimeError(f'rembg worker exited with code {proc.returncode}; see Colab log above.')
            with Image.open(dst) as result:
                return result.convert('RGBA').copy()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def worker(src, dst):
    from PIL import Image
    from rembg import new_session, remove
    print('[T4 rembg] Loading/downloading u2net, CPUExecutionProvider, 2 threads...', flush=True)
    session = new_session('u2net', providers=['CPUExecutionProvider'])
    print('[T4 rembg] Model ready; removing background...', flush=True)
    with Image.open(src) as image:
        result = remove(image.convert('RGBA'), session=session)
        result.save(dst)
    print('[T4 rembg] Complete.', flush=True)


if __name__ == '__main__':
    worker(sys.argv[1], sys.argv[2])
