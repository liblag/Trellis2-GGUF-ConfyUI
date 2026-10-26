"""CPU-only tests; no torch, model download or ComfyUI install required."""
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('execution_logging', Path(__file__).resolve().parents[1]/'execution_logging.py')
log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = []
        folders = types.ModuleType('folder_paths')
        folders.get_output_directory = lambda: self.tmp.name
        server = types.ModuleType('server')
        instance = types.SimpleNamespace(client_id='client', send_sync=lambda event, data, sid: self.events.append((event, data.copy(), sid)))
        server.PromptServer = types.SimpleNamespace(instance=instance)
        utils = types.ModuleType('comfy_execution.utils')
        utils.get_executing_context = lambda: types.SimpleNamespace(node_id='41', prompt_id='prompt')
        mgmt = types.ModuleType('comfy.model_management')
        mgmt.throw_exception_if_processing_interrupted = lambda: None
        comfy_utils = types.ModuleType('comfy.utils')
        class Bar:
            def __init__(self, total): self.total = total
            def update_absolute(self, *args): pass
        comfy_utils.ProgressBar = Bar
        self.mods = patch.dict(sys.modules, {'folder_paths': folders, 'server': server,
            'comfy_execution': types.ModuleType('comfy_execution'), 'comfy_execution.utils': utils,
            'comfy': types.ModuleType('comfy'), 'comfy.utils': comfy_utils, 'comfy.model_management': mgmt})
        self.mods.start()
        self.env = patch.dict('os.environ', {'TRELLIS_LOG_INTERVAL': '0.01'})
        self.env.start()
        self.output = contextlib.redirect_stdout(io.StringIO()); self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.env.stop(); self.mods.stop(); self.tmp.cleanup()
        self.assertIsNone(log._session.get())
        self.assertFalse(any(t.name == 'trellis-progress' for t in threading.enumerate()))

    def records(self):
        return [json.loads(line) for f in Path(self.tmp.name).rglob('*.jsonl') for line in f.read_text(encoding='utf8').splitlines()]

    def test_result_signature_nested_stage_and_heartbeat(self):
        def operation(self, number, enabled=True):
            with log.stage('decode'):
                time.sleep(.035)
            return number + enabled
        wrapped = log.wrap_node(operation, 'Example')
        self.assertEqual(inspect.signature(wrapped), inspect.signature(operation))
        self.assertEqual(wrapped(None, 3), 4)
        records = self.records()
        self.assertTrue(any(r['status'] == 'running' and r['stage'] == 'decode' for r in records))
        self.assertEqual(records[-1]['status'], 'done')
        self.assertTrue(all(r['node'] == '41' for r in records))
        self.assertTrue(all(sid == 'client' for _, _, sid in self.events))

    def test_exception_preserved_and_traceback_saved(self):
        def broken(self):
            with log.stage('kernel'):
                raise ValueError('bad kernel')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'bad kernel'):
                log.wrap_node(broken, 'Broken')(None)
        records = self.records()
        self.assertEqual(records[-1]['status'], 'error')
        self.assertIn('ValueError: bad kernel', records[-1]['traceback'])
        self.assertTrue(any(r['status'] == 'stage_error' for r in records))

    def test_sampler_progress_is_real_and_does_not_change_results(self):
        tracked = log.tracked_tqdm(lambda iterable, **kwargs: iterable)
        def operation(self):
            return [i * 2 for i in tracked(range(3), desc='sampling')]
        self.assertEqual(log.wrap_node(operation, 'Sampler')(None), [0, 2, 4])
        progress = [r for r in self.records() if r['status'] == 'progress']
        self.assertEqual([r['step'] for r in progress], [1, 2, 3])
        self.assertTrue(all(r['total'] == 3 for r in progress))

    def test_cancel_preserved(self):
        def cancelled(self): raise KeyboardInterrupt()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):log.wrap_node(cancelled, 'Cancelled')(None)
        self.assertEqual(self.records()[-1]['status'], 'error')

    def test_summary_never_reads_tensor_values(self):
        class Tensor:
            shape = (2, 3); dtype = 'float16'; device = 'cuda:0'
            def cpu(self):raise AssertionError('GPU transfer')
            def tolist(self):raise AssertionError('data read')
        self.assertEqual(log.summary(Tensor())['shape'], [2, 3])

    def test_long_pause_records_worker_stack_without_cuda_calls(self):
        session = log.Session('Slow')
        try:
            session.last_activity -= 121
            session.last_stack_dump -= 121
            session.emit('running')
        finally:
            session.close()
        record = self.records()[-1]
        self.assertIn('worker_stack', record)
        self.assertGreaterEqual(record['no_progress_seconds'], 120)
        self.assertIn('test_long_pause', record['worker_stack'])

if __name__ == '__main__': unittest.main()
