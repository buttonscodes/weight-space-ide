"""Explicit native packed CPU execution, with no dequantization fallback."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch

from .capacity import NATIVE_ROWS, NATIVE_BYTES


def availability():
    """Capability query without loading a native library or allocating weights."""
    from .runtime_config import DEFAULTS, ENV, backend_status
    import platform
    # Query only environment/portable defaults; a running serve already resolved its profile.
    try:
        settings = dict(values={**DEFAULTS, **{k:os.environ[v] for k,v in ENV.items() if k in ('backend','cpu_bridge','metal_bridge') and v in os.environ}}, platform=platform.system())
        status = backend_status(settings)
        return status['ready'], (f"Pinned ggml 0.29.0 {status['selected']} bridge configured; watchdog required"
                                if status['ready'] else '; '.join(status['blockers']) + '; see docs/RUNTIME.md and docs/QUANTIZED.md; no full-dequantization fallback')
    except ValueError as exc:
        return False, str(exc)


class PackedKernel:
    def __init__(self):
        path = os.environ.get('WSP_GGML_BRIDGE', '')
        if not path or not Path(path).is_file():
            raise RuntimeError('Packed GGUF execution needs WSP_GGML_BRIDGE; see docs/QUANTIZED.md. '
                               'Full dequantization is disabled.')
        self.build = json.loads(Path(path + '.json').read_text())
        if self.build.get('ggml') != '0.29.0' or self.build.get('abi') != 1:
            raise RuntimeError('Unsupported ggml bridge version; rebuild for ggml 0.29.0 / ABI 1.')
        self.lib = ctypes.CDLL(path)
        self.lib.wsp_abi.restype = ctypes.c_int
        if self.lib.wsp_abi() != 1:
            raise RuntimeError('Packed bridge ABI mismatch')
        self.lib.wsp_init.argtypes = [ctypes.c_char_p]
        if self.lib.wsp_init(self.build['cpu_lib'].encode()):
            raise RuntimeError('Cannot load the pinned ggml CPU library')
        self.lib.wsp_supported.argtypes = [ctypes.c_int]
        self.lib.wsp_matmul.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
        self.lib.wsp_matmul.restype = ctypes.c_int
        requested = int(os.environ.get('WSP_GGML_THREADS', '4'))
        if os.environ.get('WSP_RUNTIME_DIR'):
            from .runtime_config import runtime_thread_limit
            limit = runtime_thread_limit()
            if not 1 <= requested <= limit:
                raise ValueError(f'CPU threads must be in 1..{limit} for this host and pinned bridge')
            self.threads = requested
        else:
            self.threads = min(8, max(1, requested))  # legacy acceptance path

    def supports(self, qtype):
        return bool(self.lib.wsp_supported(int(qtype)))

    def quantize_rows(self, values, qtype):
        """Explicit export only; upstream quantizer, <=1 MiB float32 slab."""
        import gguf
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.ndim != 2 or values.size > 262144 or not np.isfinite(values).all():
            raise ValueError('Export quantizer requires finite rows of at most 262144 values')
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        if not self.supports(qtype) or values.shape[1] % block:
            raise ValueError('Unsupported export quantization shape/type')
        out = np.empty((len(values), values.shape[1]//block*size), dtype=np.uint8)
        fn = self.lib.ggml_quantize_chunk
        fn.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
        fn.restype = ctypes.c_size_t
        written = fn(int(qtype),values.ctypes.data,out.ctypes.data,0,len(values),values.shape[1],None)
        if written != out.nbytes:
            raise RuntimeError('Upstream quantizer returned an unexpected byte count')
        return out

    def matmul(self, raw, qtype, shape, x):
        import gguf
        if x.device.type != 'cpu' or not self.supports(qtype):
            raise RuntimeError(f'Packed CPU backend cannot execute {qtype} on {x.device}; no fallback.')
        rows, cols = shape
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        if x.shape[-1] != cols or cols % block or raw.shape != (rows,cols//block*size) or raw.dtype != np.uint8 or not raw.flags.c_contiguous:
            raise ValueError('Packed matmul layout mismatch')
        flat = x.detach().float().reshape(-1, cols).contiguous()
        if not len(flat):
            raise ValueError('Packed matmul requires at least one token row')
        out = torch.empty((len(flat), rows), dtype=torch.float32)
        def execute(start, count):
            rc = self.lib.wsp_matmul(int(qtype), raw.ctypes.data, rows, cols,
                                    flat[start:].data_ptr(), count, out[start:].data_ptr(), self.threads)
            # ABI 1 reports a scratch-only rejection before computation. Retry
            # smaller slabs; do not reinterpret other native failures as capacity.
            if rc == -12 and count > 1:
                half = count // 2
                execute(start, half)
                execute(start + half, count - half)
            elif rc:
                raise RuntimeError(f'ggml packed matmul failed ({rc}); no dequantization fallback')
        for start in range(0, len(flat), NATIVE_ROWS['cpu']):
            execute(start, min(NATIVE_ROWS['cpu'], len(flat)-start))
        return out.reshape(*x.shape[:-1], rows)


class MetalKernel(PackedKernel):
    """Packed mapped weights on Metal; CPU activations keep Python edit/hook semantics.

    Metal borrows the same read-only GGUF mmap. Only bounded activation input/output
    buffers are allocated per operation. Never reconstructs or copies a weight tensor.
    """
    def __init__(self, store):
        super().__init__()  # upstream CPU quantizer is used only for explicit export
        path = os.environ.get('WSP_GGML_METAL_BRIDGE','')
        build = json.loads(Path(path+'.json').read_text())
        if build.get('ggml') != '0.29.0' or build.get('abi') != 1:
            raise RuntimeError('Metal bridge version mismatch; no CPU fallback')
        self.metal = ctypes.CDLL(path)
        if self.metal.wsp_metal_abi() != 1:
            raise RuntimeError('Metal ABI mismatch')
        self.metal.wsp_metal_open.argtypes = [ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
        self.metal.wsp_metal_open.restype = ctypes.c_void_p
        self.metal.wsp_metal_close.argtypes = [ctypes.c_void_p]
        self.metal.wsp_metal_matmul.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int64,
            ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p]
        self.metal.wsp_metal_memory.argtypes = [ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64)]
        self.store = store  # owns mmap lifetime until backend is released
        self.handle = self.metal.wsp_metal_open(build['metal_lib'].encode(),store.reader.data.ctypes.data,
            store.reader.data.nbytes,max(t.n_bytes for t in store.where.values()))
        if not self.handle:
            raise RuntimeError('Cannot map packed weights on Metal; no fallback')

    def metrics(self):
        allocated, recommended = ctypes.c_uint64(), ctypes.c_uint64()
        if self.metal.wsp_metal_memory(ctypes.byref(allocated),ctypes.byref(recommended)):
            raise RuntimeError('Missing Metal allocation telemetry')
        return dict(metal_allocated=allocated.value,metal_recommended=recommended.value,
                    metal_mapped_bytes=self.store.reader.data.nbytes if self.handle else 0)

    def matmul(self, raw, qtype, shape, x):
        import gguf
        rows, cols = shape
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        if (not self.handle or not self.supports(qtype) or x.device.type != 'cpu' or x.shape[-1] != cols
            or cols % block or raw.shape != (rows,cols//block*size) or raw.dtype != np.uint8 or not raw.flags.c_contiguous):
            raise ValueError('Metal packed matmul layout/device mismatch; no fallback')
        flat = x.detach().float().reshape(-1,cols).contiguous()
        slab = min(NATIVE_ROWS['metal'], NATIVE_BYTES // (4*(rows+cols)))
        if not len(flat) or slab < 1:
            raise ValueError('A single Metal token row exceeds the bounded 128 MiB operation')
        out = torch.empty((len(flat),rows),dtype=torch.float32)
        def execute(start, count):
            status = self.metal.wsp_metal_matmul(self.handle,int(qtype),raw.ctypes.data,rows,cols,
                flat[start:].data_ptr(),count,out[start:].data_ptr())
            # Native allocation alignment can exceed the nominal byte formula.
            if status == -12 and count > 1:
                half = count // 2
                execute(start, half)
                execute(start+half, count-half)
            elif status:
                raise RuntimeError(f'Metal packed matmul failed ({status}); no fallback')
        for start in range(0, len(flat), slab):
            execute(start, min(slab, len(flat)-start))
        return out.reshape(*x.shape[:-1],rows)

    def close(self):
        if self.handle:
            self.metal.wsp_metal_close(self.handle)
            self.handle = None


def runtime_kernel(store):
    device = os.environ.get('WSP_GGML_DEVICE','cpu')
    if device == 'metal':
        return MetalKernel(store)
    if device != 'cpu':
        raise ValueError(f'Unknown packed backend: {device}')
    return PackedKernel()
