"""Explicit same-scheme GGUF merge: copy packed base, requantize edited rows.

No full tensor/model is dequantized. Existing GGUF metadata and untouched packed
bytes remain exact. The new standalone file loses patch undo and gains rounding
error; the sidecar records provenance and measured weight reconstruction error.
"""
import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import delta as D
from .gguf_store import file_hash
from .memory_guard import require_lease
from .packed import PackedKernel
from .store import norm_index


def export_merged(ws, destination):
    with ws.lock:
        return _export_merged(ws,destination)


def _export_merged(ws, destination):
    import gguf
    if ws.kind != 'gguf' or not ws.store.layout_supported:
        raise ValueError('Streamed merged GGUF needs a validated logical mapping')
    require_lease(ws)
    kernel = PackedKernel()
    target = Path(destination).resolve()
    sidecar = target.with_suffix(target.suffix+'.json')
    if target == Path(ws.store.path).resolve() or target.exists() or sidecar.exists():
        raise ValueError('Merged export requires a new path; never overwrite the base or an existing artifact')
    touched = list(ws.touched())
    for name in touched:
        shape = ws.shape(name)
        small_conv = '.linear_attn.conv1d.' in name and np.prod(shape) <= 65536 and ws.store.dtype(name) == 'F32'
        if (len(shape) not in (1,2) and not small_conv) or shape[-1] > 262144 or not kernel.supports(ws.store.where[name].tensor_type):
            raise ValueError(f'No bounded export mapping for {name}')
    if file_hash(ws.store.path) != ws.meta['base_sha256']:
        raise ValueError('Base identity changed before export')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.part')
    # Exclusive creation prevents clobbering another operation's partial file.
    error_sq, max_error, count, rows_written = 0., 0., 0, 0
    created = False
    try:
        with temporary.open('xb') as dst:
            created = True
            with open(ws.store.path,'rb') as src:
                shutil.copyfileobj(src,dst,length=1024*1024)
        reader = gguf.GGUFReader(str(temporary),mode='r+')
        tensors = {t.name:t for t in reader.tensors}
        for name in touched:
            source = ws.store.where[name]
            dest = tensors[source.name]
            shape = ws.shape(name)
            pieces = list(ws.pieces(name))
            if len(shape) != 2:
                rows = [None]
            else:
                rows = set()
                for piece, scale in pieces:
                    if not scale:
                        continue
                    r = piece.coords[:,0].tolist() if piece.kind == 'sparse' else piece.region[0]
                    rows.update([r] if isinstance(r,int) else r)
                rows = sorted(rows)
            for row in rows:
                region = norm_index((),shape) if row is None else (row,range(shape[1]))
                values = ws.store.read(name,region)
                for piece, scale in pieces:
                    D.add_to(values,region,piece,scale)
                native_values = _storage_values(ws.store,name,values.numpy())
                packed = kernel.quantize_rows(native_values.reshape(1,-1),source.tensor_type)
                reconstructed = gguf.dequantize(packed,source.tensor_type).reshape(native_values.shape)
                reconstructed = _logical_values(ws.store,name,reconstructed)
                diff = reconstructed.astype(np.float64).reshape(-1)-values.numpy().reshape(-1)
                error_sq += float(diff@diff); count += len(diff)
                max_error = max(max_error,float(np.abs(diff).max())); rows_written += 1
                if row is None:
                    dest.data.view(np.uint8).reshape(-1)[:] = packed.reshape(-1)
                else:
                    raw_row = ws.store.row_permutations.get(name,range(shape[0]))[row]
                    dest.data.view(np.uint8).reshape(shape[0],-1)[raw_row] = packed[0]
                if rows_written % 128 == 0:
                    require_lease(ws)
        reader.data.flush()
        del tensors, reader
        gc.collect()
        if file_hash(ws.store.path) != ws.meta['base_sha256']:
            raise ValueError('Base identity changed during export')
        os.link(temporary,target)  # atomic no-clobber publication, unlike replace()
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    report = dict(format='standalone-gguf-same-scheme',base_sha256=ws.meta['base_sha256'],
                  artifact_sha256=file_hash(target),path=str(target),standalone=True,
                  patches=[p.meta() for p in ws.patches if p.enabled],rows_requantized=rows_written,
                  changed_rows_max_weight_error=max_error,changed_rows_rms_weight_error=(error_sq/max(count,1))**.5,
                  float_slab_limit_bytes=1048576,backend='ggml 0.29.0 quantize_chunk',
                  limitations='Same tensor schemes only; rounding changes logits; original GGUF metadata retained; no patch undo')
    with sidecar.open('x') as f:
        json.dump(report,f,indent=2)
    return report


def _storage_values(store, name, values):
    """Bounded vector/conv or one matrix row; inverse of GGUFStore.read."""
    shape = store.shape(name)
    if store.architecture == 'qwen35' and len(shape) == 1:
        if name.endswith('.A_log'):
            with np.errstate(over='ignore', under='ignore'):
                values = -np.exp(values)
            if not np.isfinite(values).all() or not (values < 0).all():
                raise ValueError('Edited A_log is outside finite negative GGUF ssm_a range')
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name:
            values = values+1
    if len(shape) != 2 and name in store.row_permutations:
        values = values[np.argsort(store.row_permutations[name])]
    if len(shape) == 2 and name in store.column_permutations:
        values = values[np.argsort(store.column_permutations[name])]
    return values


def _logical_values(store, name, values):
    shape = store.shape(name)
    if len(shape) != 2 and name in store.row_permutations:
        values = values[store.row_permutations[name]]
    if len(shape) == 2 and name in store.column_permutations:
        values = values[store.column_permutations[name]]
    if store.architecture == 'qwen35' and len(shape) == 1:
        if name.endswith('.A_log'):
            if not (values < 0).all():
                raise ValueError('Export quantization erased a required negative ssm_a value')
            values = np.log(-values)
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name:
            values = values-1
    return values
