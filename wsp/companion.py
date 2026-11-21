"""Portable WSP companion patch bundles. Never a standalone/merged GGUF."""
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from .gguf_store import file_hash


def safe_member(name):
    """Exact portable paths; no normalization aliases or arbitrary nested files."""
    p = PurePosixPath(name)
    if '\\' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        return False
    if name in ('workspace.json','groups.json','task.json','corpus.txt','notes.md'):
        return True
    if name == 'model/chat_template.jinja':
        return True
    if len(p.parts) == 3 and p.parts[0] == 'model' and p.parts[1] in ('chat_templates','additional_chat_templates'):
        return p.suffix == '.jinja' and bool(p.stem)
    if name in ('model/tokenizer.model','model/spiece.model','model/vocab.txt','model/merges.txt'):
        return True
    return len(p.parts) == 2 and p.parts[0] in ('patches','model','tests') and p.suffix in ('.json','.safetensors')


def export_bundle(ws, path):
    if getattr(ws, "_runtime_client", None) is not None:
        ws._runtime_client.require(ws)
    if file_hash(ws.store.path) != ws.meta['base_sha256']:
        raise ValueError('GGUF base identity changed; refusing export')
    path = str(Path(path).with_suffix('.wsppatch.zip'))
    if Path(path).resolve() == Path(ws.store.path).resolve():
        raise ValueError('Cannot overwrite the immutable base')
    files = ['workspace.json']
    for name in ('groups.json','task.json','corpus.txt','notes.md'):
        if Path(ws.dir,name).is_file():
            files.append(name)
    for directory in ('model', 'model/chat_templates', 'model/additional_chat_templates', 'patches','tests'):
        for p in sorted(Path(ws.dir, directory).glob('*')):
            name = p.relative_to(ws.dir).as_posix()
            if p.is_file() and safe_member(name):
                files.append(name)
    for name in files:
        if any(Path(ws.dir, *Path(name).parts[:i]).is_symlink() for i in range(1, len(Path(name).parts)+1)):
            raise ValueError(f'Symlink in companion source: {name}')
    if len(files) >= 10000 or sum(Path(ws.dir, f).stat().st_size for f in files) > 256*1024**2:
        raise ValueError('Companion bundle exceeds the 256 MiB patch limit')
    manifest = dict(format='wsp-gguf-companion-v1', base_sha256=ws.meta['base_sha256'],
                    base_path=ws.store.path, representation='reconstructed quantized base + FP32 deltas',
                    standalone=False, files={f: file_hash(Path(ws.dir, f)) for f in files})
    with zipfile.ZipFile(path+'.part', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        for f in files:
            z.write(Path(ws.dir, f), f)
    os.replace(path+'.part', path)
    return dict(path=path, format=manifest['format'], standalone=False, base_sha256=manifest['base_sha256'])


def import_bundle(root, path, base_path=None, metadata_only=False):
    from .workspace import Workspace
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > 256*1024**2 or len(z.infolist()) > 10000:
            raise ValueError('Companion bundle exceeds import limits')
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError('Duplicate companion member')
        manifest = json.loads(z.read('manifest.json'))
        if manifest.get('format') != 'wsp-gguf-companion-v1':
            raise ValueError('Not a WSP GGUF companion bundle')
        base = str(Path(base_path or manifest['base_path']).resolve())
        if not metadata_only and file_hash(base) != manifest['base_sha256']:
            raise ValueError('Companion base SHA-256 mismatch; select the exact original GGUF')
        payload = {}
        for name, digest in manifest['files'].items():
            if not safe_member(name):
                raise ValueError(f'Unsafe companion member: {name}')
            data = z.read(name)
            import hashlib
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Companion checksum mismatch: {name}')
            payload[name] = data
        meta = json.loads(payload['workspace.json'])
        if meta.get('kind') != 'gguf' or meta.get('base_sha256') != manifest['base_sha256']:
            raise ValueError('Companion workspace/base identity mismatch')
        os.makedirs(root, exist_ok=True)
        directory = tempfile.mkdtemp(prefix='companion-', dir=root)
        for name, data in payload.items():
            dest = Path(directory, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        meta.update(gguf_path=base, model_dir='model', name=Path(directory).name)
        Path(directory, 'workspace.json').write_text(json.dumps(meta, indent=2))
    return Workspace.load(directory, metadata_only=metadata_only)
