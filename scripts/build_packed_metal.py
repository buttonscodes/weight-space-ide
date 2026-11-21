"""Optional macOS Metal bridge, reusing an isolated ggml 0.29.0 build."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--prefix',required=True)
p.add_argument('--metal-lib',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
prefix, out = Path(a.prefix).resolve(), Path(a.out).resolve()
if 'Version: 0.29.0\n' not in (prefix/'lib/pkgconfig/ggml.pc').read_text():
    p.error('Requires pinned ggml 0.29.0')
source = Path(__file__).resolve().parents[1]/'native/packed_metal.mm'
out.parent.mkdir(parents=True,exist_ok=True)
cmd = ['c++','-std=c++17','-O2','-fPIC','-shared',str(source),'-I'+str(prefix/'include'),
       '-L'+str(prefix/'lib'),'-lggml-base','-framework','Metal','-framework','Foundation',
       '-Wl,-rpath,'+str(prefix/'lib'),'-o',str(out)]
subprocess.run(cmd,check=True)
Path(str(out)+'.json').write_text(json.dumps(dict(ggml='0.29.0',abi=1,command=cmd,
    metal_lib=str(Path(a.metal_lib).resolve())),indent=2))
print(f'export WSP_GGML_METAL_BRIDGE={out}')
