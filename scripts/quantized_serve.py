"""Bounded browser acceptance server with explicit workspace and owned stop-file."""
import argparse
from pathlib import Path
import threading
import time

import torch
import uvicorn
from wsp.server import create_app
from wsp.lifecycle import IDEServer
from wsp.session import Session

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--root', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--stop-file', required=True)
a = p.parse_args()
if Path(a.stop_file).exists():
    p.error('Stop file already exists; choose a fresh validation run')
torch.set_num_threads(4)
s = Session(a.root)
s.open(a.model)
server = IDEServer(uvicorn.Config(create_app(s,f'http://127.0.0.1:{a.port}'),host='127.0.0.1',port=a.port,
                                       log_level='warning',timeout_graceful_shutdown=3))
def stop_monitor():
    while not server.should_exit:
        if Path(a.stop_file).exists():
            server.should_exit = True
            return
        time.sleep(1)
threading.Thread(target=stop_monitor,daemon=True).start()
try:
    server.run()
finally:
    if s.ws and s.ws._runner:
        s.ws._runner.release()
        s.ws._runner = None
