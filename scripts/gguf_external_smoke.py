"""Bounded pinned llama.cpp CPU portability probe; launch ONLY inside memory_guard.

CPU-only build avoids unobservable external Metal allocations. This is independent
file/runtime portability evidence, not a substitute for the IDE fixed-suite gain gate.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import resource
import platform
import time

from wsp.gguf_store import open_gguf, file_hash
from wsp.memory_guard import require_lease, stage

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--binary',required=True)
p.add_argument('--out',required=True);p.add_argument('--prompt',default='The capital of France is')
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
w=open_gguf(str(out/'metadata'),a.model)
require_lease(w,loading=True)
identity=dict(artifact_sha256=w.meta['base_sha256'],parameters=w.n_params(),tensor_types=w.store.capabilities()['tensor_types'])
if len(w.encode(a.prompt))+4>64:raise ValueError('Prompt exceeds bounded64-token context')
del w;gc.collect()
# The pinned external build is CPU-only: no hidden Metal allocator. Inspect the
# CMakeCache before spawn, and include it in the provenance record.
cache=Path(a.binary).resolve().parents[1]/'CMakeCache.txt'
if 'GGML_METAL:BOOL=OFF' not in cache.read_text():raise ValueError('External smoke requires pinned CPU-only build for allocation telemetry')
command=[a.binary,'-m',a.model,'-p',a.prompt,'-n','4','-c','64','-b','64','-ub','64',
         '-t','4','-tb','4','-ngl','0','--seed','0','--temp','0','-no-cnv','--no-warmup','--no-display-prompt','--simple-io']
stage('external four-token smoke',60);started=time.monotonic()
with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
    result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=55,check=False)
report=dict(**identity,command=command,returncode=result.returncode,seconds=time.monotonic()-started,
            prompt=a.prompt,text=(out/'stdout.txt').read_text(),backend='llama.cpp CPU only; no Metal',
            external_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
            binary_sha256=file_hash(a.binary),cmake_cache_sha256=file_hash(cache),
            qualification='Raw completion, no chat template; CPU arithmetic may differ from IDE Metal; no edit-equivalence claim from token agreement')
(out/'report.json').write_text(json.dumps(report,indent=2))
if result.returncode:raise RuntimeError(f'External runtime failed: {result.returncode}')
