"""Explicit standalone merge, under the same external watchdog as inference."""
import argparse
import json
from wsp.gguf_mixed import export_mixed, preview_export
from wsp.workspace import Workspace

p = argparse.ArgumentParser()
p.add_argument('--workspace',required=True)
p.add_argument('--out',required=True,help='New standalone GGUF path; original base is never overwritten')
p.add_argument('--precision',choices=['SAME','Q8_0','F16','F32'],default='F16')
p.add_argument('--preview',action='store_true',help='Metadata-only plan; no watchdog or inference needed')
a = p.parse_args()
w = Workspace.load(a.workspace, metadata_only=a.preview)
print(json.dumps(preview_export(w,a.precision) if a.preview else export_mixed(w,a.out,a.precision),indent=2))
