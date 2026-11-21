"""Build the optional, version-pinned ggml bridge; never installs system packages."""
import argparse
import json
import pathlib
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--prefix', required=True, help='ggml 0.29.0 installation prefix')
p.add_argument('--cpu-lib', required=True, help='absolute path to the ggml CPU shared library')
p.add_argument('--out', required=True, help='output shared library (outside source control)')
a = p.parse_args()
prefix = pathlib.Path(a.prefix).resolve()
pc = (prefix / 'lib/pkgconfig/ggml.pc').read_text()
if 'Version: 0.29.0\n' not in pc:
    p.error('This bridge is validated against ggml 0.29.0; rebuild/revalidate for other versions.')
out = pathlib.Path(a.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
src = pathlib.Path(__file__).resolve().parents[1] / 'native/packed.cpp'
cmd = ['c++', '-std=c++17', '-O2', '-fPIC', '-shared', str(src), '-I'+str(prefix/'include'),
       '-L'+str(prefix/'lib'), '-lggml-base',
       '-Wl,-rpath,'+str(prefix/'lib'), '-o', str(out)]
subprocess.run(cmd, check=True)
out.with_suffix(out.suffix+'.json').write_text(json.dumps({'ggml': '0.29.0', 'abi': 1, 'command': cmd,
    'cpu_lib': str(pathlib.Path(a.cpu_lib).resolve())}, indent=2))
print(f'export WSP_GGML_BRIDGE={out}')
