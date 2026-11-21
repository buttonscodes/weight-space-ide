"""Hugging Face Hub: find a model's files, fetch the small ones, leave the weights where they are.

    resolve("Qwen/Qwen2.5-0.5B-Instruct")  ->  config, tokenizer files (downloaded, a few MB)
                                               and the URLs of the safetensors shards (not downloaded)

The weight files are then read lazily through `store.Store` (HTTP range requests), or pulled to
disk with `pull` when the model is going to be run.
"""
from __future__ import annotations

import json
import os
import urllib.request

API = "https://huggingface.co/api/models/{repo}/revision/{rev}"
URL = "https://huggingface.co/{repo}/resolve/{rev}/{file}"
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json", "chat_template.jinja", "chat_template.json",
         "model.safetensors.index.json")


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url)
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        tmp = path + ".part"
        with urllib.request.urlopen(req, timeout=3600) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, path)
    return path


def resolve(repo: str, directory: str, rev: str = "main") -> dict:
    """Download config and tokenizer files into `directory`; return {config, weights: [urls], files, sha}."""
    info = json.loads(_get(API.format(repo=repo, rev=rev)))
    files = [s["rfilename"] for s in info.get("siblings", [])]
    sha = info.get("sha") or rev
    os.makedirs(directory, exist_ok=True)
    for f in files:
        if f in SMALL:
            fetch(URL.format(repo=repo, rev=sha, file=f), os.path.join(directory, f))
    config = {}
    cp = os.path.join(directory, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            config = json.load(f)
    ip = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(ip):
        with open(ip) as f:
            weights = sorted(set(json.load(f)["weight_map"].values()))
    elif "model.safetensors" in files:
        weights = ["model.safetensors"]
    else:
        weights = sorted(f for f in files if f.endswith(".safetensors") and "/" not in f)
    if not weights:
        raise FileNotFoundError(f"{repo} has no safetensors weights (found: {[f for f in files if f.endswith(('.bin', '.gguf', '.pt'))][:5]}). "
                                f"Convert the checkpoint to safetensors first.")
    return {"repo": repo, "sha": sha, "config": config, "weights": weights,
            "urls": [URL.format(repo=repo, rev=sha, file=w) for w in weights], "files": files}


def pull(repo: str, sha: str, weights: list[str], directory: str, log=print) -> list[str]:
    """Download the weight files next to the config (needed to run the model; not needed to read or edit it)."""
    out = []
    for w in weights:
        path = os.path.join(directory, w)
        if not os.path.exists(path):
            log(f"downloading {w}")
        out.append(fetch(URL.format(repo=repo, rev=sha, file=w), path))
    return out
