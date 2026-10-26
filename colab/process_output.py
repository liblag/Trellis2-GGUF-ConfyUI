"""Stream subprocess output into notebook stdout and a persistent text log."""
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def run_streamed(args, log_path, timeout=None, env=None):
    args = list(map(str, args))
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ if env is None else env, PYTHONUNBUFFERED='1')
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf8', errors='replace', bufsize=1,
                               env=environment, start_new_session=os.name != 'nt')
    output = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                output.put(line)
        finally:
            output.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    start = time.monotonic()
    try:
        with log_path.open('a', encoding='utf8', buffering=1) as log:
            log.write('\nCOMMAND: '+' '.join(args)+'\n')
            ended = False
            while not ended:
                if timeout is not None and time.monotonic()-start > timeout:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    line = output.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                else:
                    print(line, end='', flush=True)
                    log.write(line)
            code = process.wait(timeout=5)
            log.write(f'EXIT CODE: {code}\n')
        if code:
            print(f'Full subprocess log: {log_path}', flush=True)
            raise subprocess.CalledProcessError(code, args)
        return subprocess.CompletedProcess(args, code)
    finally:
        if process.poll() is None:
            if os.name == 'nt': process.terminate()
            else: os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'nt': process.kill()
                else: os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()
