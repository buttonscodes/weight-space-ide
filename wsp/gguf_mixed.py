"""Standard GGUF mixed-precision export, with bounded logical reconstruction.

Preview only reads tensor/patch metadata. Export requires the existing external
lease, but never loads a runner. Untouched payloads are copied verbatim; a touched
matrix is promoted in its entirety, one row at a time. Behavioral status starts
UNVALIDATED regardless of successful serialization or low weight error.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time

import gguf
import numpy as np

from . import delta as D
from .gguf_export import _logical_values, _storage_values
from .gguf_store import SCHEMES
from .memory_guard import require_lease
from .packed import PackedKernel
from .store import norm_index

SLAB = 1024 * 1024
MODES = ('SAME', 'Q8_0', 'F16', 'F32')


class ExportCancelled(RuntimeError):
    pass


class _Checkpoint:
    def __init__(self, ws, cancel):
        self.ws, self.cancel, self.last = ws, cancel, float('-inf')

    def __call__(self):
        if self.cancel is not None and self.cancel.is_set():
            raise ExportCancelled('Export cancelled; no final artifact published')
        if time.monotonic() - self.last >= .5:
            require_lease(self.ws)
            self.last = time.monotonic()


def _type(source, requested):
    q = gguf.GGMLQuantizationType
    if requested not in MODES:
        raise ValueError('Precision must be SAME, Q8_0, F16 or F32')
    if requested == 'SAME':
        return source
    if source == q.F32:
        return source
    if source == q.BF16:
        # F16 loses BF16 range; Q8 also loses precision. F32 covers both.
        return source if requested == 'Q8_0' else q.F32
    if source == q.F16 and requested == 'Q8_0':
        return source
    return getattr(q, requested)


def _string(value):
    value = value.encode('utf-8')
    return struct.pack('<Q', len(value)) + value


def _metadata(ws, precision):
    reader = ws.store.reader
    # File type is optional in GGUF. Remove the old uniform/mixed quantization
    # label rather than mislabelling this file MOSTLY_F16 or uniform Q4.
    replacements = {'wsp.export.precision': precision, 'wsp.export.base_sha256': ws.meta['base_sha256'],
                    'wsp.export.validation': 'UNVALIDATED',
                    'wsp.export.representation': 'standard per-tensor mixed precision; edits baked in; no undo'}
    raw = [f for k, f in reader.fields.items() if not k.startswith('GGUF.')
           and k != 'general.file_type' and k not in replacements]
    added = [_string(k) + struct.pack('<I', int(gguf.GGUFValueType.STRING)) + _string(v)
             for k, v in replacements.items()]
    return raw, added


def _plan(ws, precision):
    if ws.kind != 'gguf' or not ws.store.layout_supported:
        raise ValueError('Mixed export needs a validated GGUF logical mapping')
    if precision not in MODES:
        raise ValueError('Precision must be SAME, Q8_0, F16 or F32')
    touched = {n for n in ws.touched() if any(scale and not p.is_zero() for p, scale in ws.pieces(n))}
    reverse = {t.name: n for n, t in ws.store.where.items()}
    entries = []
    alignment = int(ws.store.reader.alignment)
    if alignment > SLAB:raise ValueError('GGUF alignment exceeds bounded export limit')
    align = lambda n: (n + alignment - 1) // alignment * alignment
    offset = 0
    for t in ws.store.reader.tensors:
        name = reverse[t.name]
        changed = name in touched
        shape = ws.shape(name)
        if changed and t.tensor_type.name not in SCHEMES:
            raise ValueError(f'No logical decoder for edited tensor {name}: {t.tensor_type.name}')
        if changed and (len(shape) not in (1, 2) and not ('.linear_attn.conv1d.' in name and t.n_elements <= 65536)):
            raise ValueError(f'No bounded export mapping for {name}')
        if changed and (shape[-1] * 4 > SLAB or (len(shape) != 2 and t.n_elements * 4 > SLAB)):
            raise ValueError(f'{name}: row/vector exceeds 1MiB reconstruction limit')
        dtype = _type(t.tensor_type, precision) if changed else t.tensor_type
        block, size = gguf.GGML_QUANT_SIZES[dtype]
        if changed and int(t.shape[0]) % block:
            raise ValueError(f'{name}: innermost dimension is not divisible by {dtype.name} block {block}; choose F16/F32')
        nbytes = int(t.n_elements) // block * size
        entries.append(dict(name=name, tensor=t, changed=changed, dtype=dtype, offset=offset, nbytes=nbytes))
        offset += align(nbytes)
    raw, added = _metadata(ws, precision)
    header = 24 + sum(sum(p.nbytes for p in f.parts) for f in raw) + sum(map(len, added))
    header += sum(8 + len(e['tensor'].name.encode()) + 4 + 8 * len(e['tensor'].shape) + 4 + 8 for e in entries)
    start = align(header)
    return entries, raw, added, start, start + offset


def preview_export(ws, precision='F16'):
    with ws.lock:
        entries, _, _, start, total = _plan(ws, precision)
        changes = [dict(name=e['name'], gguf_name=e['tensor'].name, shape=list(ws.shape(e['name'])),
                        before=e['tensor'].tensor_type.name, after=e['dtype'].name,
                        before_bytes=int(e['tensor'].n_bytes), after_bytes=e['nbytes']) for e in entries if e['changed']]
        original = os.path.getsize(ws.store.path)
        return dict(mode='same-scheme' if precision=='SAME' else 'mixed', precision=precision, validation_status='UNVALIDATED', base_sha256=ws.meta['base_sha256'],
                    changed_tensor_count=len(changes), tensors=changes, output_bytes=total,
                    original_bytes=original, increase_bytes=total-original, data_offset=start,
                    float_slab_limit_bytes=SLAB, requires_whole_tensor_promotion=True,
                    limitations='A tiny patch promotes its whole tensor; no undo; no inference in preview/export; use companion for faithful edits')


def export_mixed(ws, destination, precision='F16', *, cancel=None, progress=None):
    with ws.lock:
        return _export(ws, destination, precision, cancel, progress)


def _export(ws, destination, precision, cancel, progress):
    entries, fields, added, start, total = _plan(ws, precision)
    check = _Checkpoint(ws, cancel)
    check()
    target = Path(destination).resolve(); sidecar = Path(str(target)+'.json')
    if target == Path(ws.store.path).resolve() or target.exists() or sidecar.exists():
        raise ValueError('Mixed export requires a new path; destination or report already exists')
    if not target.parent.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < total + 16*SLAB:
        raise ValueError(f'Insufficient disk: need {total+16*SLAB} free bytes for export and report')
    # Hashing is streaming and cancellable too; no whole tensor touched here.
    def hashed(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(SLAB):
                check(); h.update(chunk)
        return h.hexdigest()
    if hashed(ws.store.path) != ws.meta['base_sha256']:
        raise ValueError('Base identity changed before export')
    kernel = PackedKernel()
    fd, temporary_name = tempfile.mkstemp(prefix='.'+target.name+'-', suffix='.part', dir=target.parent)
    temporary = Path(temporary_name); report_tmp = Path(str(temporary)+'.json')

    max_error = sq = 0.; count = rows = 0
    try:
        with os.fdopen(fd, 'w+b') as dst, open(ws.store.path, 'rb') as src:
            def copy(offset, length):
                src.seek(offset)
                while length:
                    check()
                    chunk = src.read(min(length, SLAB))
                    if not chunk: raise ValueError('Source truncated during export')
                    dst.write(chunk); length -= len(chunk)
            dst.write(struct.pack('<4sIQQ', b'GGUF', 3, len(entries), len(fields)+len(added)))
            for f in fields: copy(f.offset, sum(p.nbytes for p in f.parts))
            for value in added: dst.write(value)
            for e in entries:
                t=e['tensor']
                dst.write(_string(t.name)+struct.pack('<I',len(t.shape)))
                dst.write(struct.pack('<'+'Q'*len(t.shape),*map(int,t.shape)))
                dst.write(struct.pack('<IQ',int(e['dtype']),e['offset']))
            dst.write(b'\0'*(start-dst.tell()))
            for e in entries:
                check()
                t=e['tensor']; name=e['name']; shape=ws.shape(name)
                if progress: progress(dict(tensor=t.name, changed=e['changed'], bytes_written=dst.tell(), total_bytes=total))
                dst.seek(start+e['offset'])
                if not e['changed']:
                    copy(t.data_offset,t.n_bytes)
                else:
                    pieces=list(ws.pieces(name)); row_bytes=e['nbytes']//shape[0] if len(shape)==2 else e['nbytes']
                    for row in range(shape[0]) if len(shape)==2 else [None]:
                        check()
                        region=(row,range(shape[1])) if row is not None else norm_index((),shape)
                        values=ws.store.read(name,region)
                        for piece,scale in pieces: D.add_to(values,region,piece,scale)
                        native=_storage_values(ws.store,name,values.numpy())
                        if not np.isfinite(native).all(): raise ValueError(f'Nonfinite edited values in {name}')
                        encoded=kernel.quantize_rows(native.reshape(1,-1),e['dtype'])
                        decoded=_logical_values(ws.store,name,gguf.dequantize(encoded,e['dtype']).reshape(native.shape))
                        if not np.isfinite(decoded).all(): raise ValueError(f'{e["dtype"].name} overflow in {name}; choose F32')
                        diff=decoded.astype(np.float64).reshape(-1)-values.numpy().reshape(-1)
                        max_error=max(max_error,float(np.abs(diff).max()));sq+=float(diff@diff);count+=diff.size;rows+=1
                        native_row=ws.store.row_permutations.get(name,range(shape[0]))[row] if row is not None else 0
                        dst.seek(start+e['offset']+int(native_row)*row_bytes)
                        dst.write(encoded.tobytes())
            dst.truncate(total); dst.flush();os.fsync(dst.fileno())
        if hashed(ws.store.path)!=ws.meta['base_sha256']: raise ValueError('Base identity changed during export')
        report=dict(preview_export(ws,precision),format='standalone-gguf-same-scheme' if precision=='SAME' else 'standalone-gguf-mixed',standalone=True,path=str(target),
                    artifact_sha256=hashed(temporary),
                    rows_encoded=rows,changed_rows_max_weight_error=max_error,
                    changed_rows_rms_weight_error=(sq/max(count,1))**.5,
                    backend='ggml 0.29.0 quantize_chunk',structural_status='PASSED',
                    patches=[p.meta() for p in ws.patches if p.enabled and p.strength])
        with report_tmp.open('x') as f: json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
        check()
        # Report precedes artifact publication. Roll back only our own report if
        # another writer wins the artifact race. No incomplete final GGUF exposed.
        os.link(report_tmp,sidecar)
        try: os.link(temporary,target)
        except BaseException:
            if sidecar.stat().st_ino==report_tmp.stat().st_ino: sidecar.unlink()
            raise
        return report
    finally:
        temporary.unlink(missing_ok=True);report_tmp.unlink(missing_ok=True)
