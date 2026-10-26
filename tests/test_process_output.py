import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('process_output',Path(__file__).resolve().parents[1]/'colab/process_output.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class OutputTests(unittest.TestCase):
    def test_stdout_stderr_are_visible_and_saved(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            path=Path(tmp)/'build.log'
            mod.run_streamed([sys.executable,'-c',"import sys;print('stdout');print('compiler error',file=sys.stderr)"],path)
            self.assertIn('compiler error',output.getvalue())
            self.assertIn('stdout',path.read_text())

    def test_error_keeps_actual_cause(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path=Path(tmp)/'build.log'
            with self.assertRaises(subprocess.CalledProcessError):
                mod.run_streamed([sys.executable,'-c',"import sys;print('unsupported architecture');sys.exit(3)"],path)
            self.assertIn('unsupported architecture',path.read_text())
            self.assertIn('EXIT CODE: 3',path.read_text())

    def test_silent_child_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.TimeoutExpired):
                mod.run_streamed([sys.executable,'-c','import time;time.sleep(10)'],Path(tmp)/'log',timeout=.1)

if __name__=='__main__':unittest.main()
