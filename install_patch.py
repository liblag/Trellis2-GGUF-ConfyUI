"""Apply the background worker patch without overwriting unrelated nodes.py edits."""
import ast
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv)>1 else '/content/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF')
path = root/'nodes.py'
text = path.read_text(encoding='utf8')
old = '            from rembg import remove\n            image = remove(image)'
new = '            from .t4_background import remove_background as remove\n            image = remove(image)'
if new not in text:
    if text.count(old) != 1:
        raise RuntimeError('PreProcess source differs from expected version; no changes made.')
    patched = text.replace(old, new)
    ast.parse(patched)
    backup = root/'nodes.py.before-t4-background'
    if not backup.exists():shutil.copy2(path, backup)
    shutil.copy2(Path(__file__).with_name('t4_background.py'), root/'t4_background.py')
    path.write_text(patched,encoding='utf8')
else:
    shutil.copy2(Path(__file__).with_name('t4_background.py'), root/'t4_background.py')
print('Background worker patch installed. Restart ComfyUI to activate.')
