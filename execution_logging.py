"""Per-node stage logs and UI events. Does not synchronize or poll CUDA."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid

_session = ContextVar('trellis_log_session', default=None)
EVENT = 'trellis2-stage'


def summary(value):
    """Never materialize tensors, read GPU values, or dump image/prompt contents."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:160] if len(value) <= 160 else f'string({len(value)} chars)'
    shape = getattr(value, 'shape', None)
    if shape is not None:
        return {'type': type(value).__name__, 'shape': list(shape),
                'dtype': str(getattr(value, 'dtype', '')), 'device': str(getattr(value, 'device', ''))}
    if isinstance(value, dict):
        return {str(k): summary(v) for k, v in list(value.items())[:16]}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'length': len(value), 'items': [summary(x) for x in value[:4]]}
    if hasattr(value, 'size') and isinstance(value.size, tuple):
        return {'type': type(value).__name__, 'size': value.size}
    return {'type': type(value).__name__}


class Session:
    def __init__(self, node_type, interval=None):
        self.node_type = node_type
        self.node_id = None
        self.prompt_id = None
        try:
            from comfy_execution.utils import get_executing_context
            context = get_executing_context()
            if context:
                self.node_id, self.prompt_id = context.node_id, context.prompt_id
        except ImportError:
            pass
        self.server = None
        self.sid = None
        try:
            from server import PromptServer
            self.server = PromptServer.instance
            self.sid = self.server.client_id
        except (ImportError, AttributeError):
            pass
        self.started = time.monotonic()
        self.worker_thread = threading.get_ident()
        self.last_activity = self.started
        self.last_stack_dump = self.started
        self.stack = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.interval = interval if interval is not None else float(os.environ.get('TRELLIS_LOG_INTERVAL', '20'))
        self.file = None
        try:
            import folder_paths
            directory = Path(folder_paths.get_output_directory())/'trellis_logs'
            directory.mkdir(parents=True, exist_ok=True)
            node = re.sub(r'[^a-zA-Z0-9_-]', '_', str(self.node_id or 'unknown'))
            name = time.strftime('%Y%m%d-%H%M%S')+f'-node-{node}-{uuid.uuid4().hex[:8]}.jsonl'
            self.file = (directory/name).open('a', encoding='utf8', buffering=1)
        except (ImportError, OSError):
            pass

    def emit(self, status, **extra):
        with self.lock:
            now = time.monotonic()
            if status != 'running':
                self.last_activity = now
            frame = self.stack[-1] if self.stack else None
            data = {'node': self.node_id, 'prompt_id': self.prompt_id, 'node_type': self.node_type,
                    'status': status, 'stage': ' / '.join(x['name'] for x in self.stack) or self.node_type,
                    'elapsed': round(now-self.started, 1),
                    'stage_elapsed': round(now-frame['started'], 1) if frame else 0,
                    'step': frame.get('step') if frame else None, 'total': frame.get('total') if frame else None,
                    **extra}
            if status == 'running':
                data['no_progress_seconds'] = round(now-self.last_activity, 1)
                if now-self.last_activity >= 120 and now-self.last_stack_dump >= 120:
                    frame_info = sys._current_frames().get(self.worker_thread)
                    if frame_info is not None:
                        data['worker_stack'] = ''.join(traceback.format_stack(frame_info))
                    self.last_stack_dump = now
            if status == 'running':
                # CPU/RAM only: CUDA queries on the heartbeat thread could themselves block.
                try:
                    import psutil
                    data['rss_gb'] = round(psutil.Process().memory_info().rss/1024**3, 2)
                    data['ram_available_gb'] = round(psutil.virtual_memory().available/1024**3, 2)
                except (ImportError, OSError):
                    pass
            count = f" {data['step']}/{data['total']}" if data['total'] else ''
            print(f"[Trellis][node={self.node_id}][{status.upper()}] {data['stage']}{count} | {data['elapsed']}s total | {data['stage_elapsed']}s stage"+
                  (f" | RAM {data['rss_gb']}GB, free {data['ram_available_gb']}GB" if 'rss_gb' in data else ''), flush=True)
            if status in ('inputs', 'done'):
                field = 'inputs' if status == 'inputs' else 'outputs'
                print('[Trellis] '+field+': '+json.dumps(data.get(field), ensure_ascii=False, default=str)[:2000], flush=True)
            if 'worker_stack' in data:
                print(f"[Trellis] No completed step/stage for {data['no_progress_seconds']}s. Worker Python stack (not proof of a deadlock):\n"+data['worker_stack'], flush=True)
            if self.file:
                try:
                    self.file.write(json.dumps(data, ensure_ascii=False, default=str)+'\n')
                except OSError:
                    pass
            if self.server is not None and self.node_id is not None:
                try:
                    self.server.send_sync(EVENT, data, self.sid)
                except Exception:
                    pass  # UI delivery must never break inference.

    def start(self):
        self.emit('start')
        def heartbeat():
            while not self.stop_event.wait(self.interval):
                self.emit('running')
        self.thread = threading.Thread(target=heartbeat, name='trellis-progress', daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None


@contextmanager
def stage(name, total=None):
    session = _session.get()
    if session is None:
        yield
        return
    with session.lock:
        session.stack.append({'name': name, 'started': time.monotonic(), 'step': 0 if total else None, 'total': total})
        session.emit('stage_start')
    try:
        yield
    except BaseException as exc:
        session.emit('stage_error', error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        session.emit('stage_done')
    finally:
        with session.lock:
            session.stack.pop()


def step_progress(current, total):
    session = _session.get()
    if session is None:
        return
    with session.lock:
        if session.stack:
            session.stack[-1].update(step=current, total=total)
        session.emit('progress')


def tracked_tqdm(original):
    @wraps(original)
    def tracked(iterable=None, *args, **kwargs):
        # Existing callers using tqdm(total=...) keep the original manual interface.
        if iterable is None or _session.get() is None:
            return original(iterable, *args, **kwargs)
        def iterate():
            total = kwargs.get('total')
            if total is None:
                try:
                    total = len(iterable)
                except TypeError:
                    pass
            with stage(kwargs.get('desc') or 'Iterations', total=total):
                progress = None
                if total:
                    try:
                        from comfy.utils import ProgressBar
                        progress = ProgressBar(total)
                    except ImportError:
                        pass
                for index, value in enumerate(original(iterable, *args, **kwargs), 1):
                    # Checks cancellation on the inference thread, not the heartbeat thread.
                    from comfy.model_management import throw_exception_if_processing_interrupted
                    throw_exception_if_processing_interrupted()
                    yield value
                    step_progress(index, total)
                    if progress:
                        progress.update_absolute(index, total)
        return iterate()
    return tracked


def wrapped_stage(function, label):
    if getattr(function, '_trellis_stage_log', False):
        return function
    @wraps(function)
    def logged(*args, **kwargs):
        with stage(label):
            return function(*args, **kwargs)
    logged._trellis_stage_log = True
    return logged


def wrap_node(function, node_type):
    if getattr(function, '_trellis_node_log', False):
        return function
    @wraps(function)
    def logged(*args, **kwargs):
        if _session.get() is not None:
            return function(*args, **kwargs)
        session = Session(node_type)
        token = _session.set(session)
        session.start()
        try:
            # Only data structure summaries; no full tensors or image arrays.
            session.emit('inputs', inputs=summary({'args': args[1:], 'kwargs': kwargs}))
            result = function(*args, **kwargs)
            session.emit('done', outputs=summary(result))
            return result
        except BaseException as exc:
            session.emit('error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            traceback.print_exc()
            raise
        finally:
            session.close()
            _session.reset(token)
    logged._trellis_node_log = True
    return logged


def install_execution_logging(mapping):
    # Every registered Trellis node, including load/export/remesh/UV/texturing.
    for node_type, cls in mapping.items():
        method = getattr(cls, 'FUNCTION', None)
        if method:
            setattr(cls, method, wrap_node(getattr(cls, method), node_type))
    from .trellis2_gguf.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_gguf.pipelines import trellis2_texturing
    for cls in [Trellis2ImageTo3DPipeline, *[x for x in vars(trellis2_texturing).values()
                if inspect.isclass(x) and x.__module__ == trellis2_texturing.__name__]]:
        for name, method in list(vars(cls).items()):
            if inspect.isfunction(method) and name.startswith(('load_', 'unload_', 'sample_', 'decode_', 'get_cond', 'get_proj_cond')):
                setattr(cls, name, wrapped_stage(method, name))
    from .trellis2_gguf.pipelines.samplers import flow_euler
    if not getattr(flow_euler.tqdm, '_trellis_tracked', False):
        flow_euler.tqdm = tracked_tqdm(flow_euler.tqdm)
        flow_euler.tqdm._trellis_tracked = True
    from .trellis2_gguf.models.sc_vaes import fdg_vae
    for name in ['flexible_dual_grid_to_mesh', 'tiled_flexible_dual_grid_to_mesh']:
        setattr(fdg_vae, name, wrapped_stage(getattr(fdg_vae, name), name))
