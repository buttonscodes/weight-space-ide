"""PEFT's own merge of a LoRA adapter into its base, saved as a checkpoint: the oracle of the IDE's adapter input (H4b).

Run by the orchestrator with Heretic's interpreter (it has peft 0.21.2; it loads the base model), offline, under the
memory guard, one process at a time:

    python scripts/peft_merge_adapter.py <base snapshot dir> <adapter dir> <out dir>

The base is loaded with AutoModelForCausalLM.from_pretrained(<base>, dtype=torch.float32) (the dtype of Heretic's
float32 comparison runs), the adapter attached with PeftModel.from_pretrained(model, <adapter>) (its defaults, among
them autocast_adapter_dtype), merged with merge_and_unload() and saved with save_pretrained(<out>) (safetensors);
the base's tokenizer is saved beside it with AutoTokenizer.save_pretrained, as Heretic saves a merged model
(heretic/main.py). <out> must not exist or be empty. <out>/peft_merge.json records the versions, the adapter
file's SHA-256 and every warning PEFT raised. Nothing here is a reimplementation of PEFT.
"""
import hashlib
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path


def main(argv):
    if len(argv) != 3 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    base, adapter, out = (Path(a) for a in argv)
    if out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty; give a fresh directory", file=sys.stderr)
        return 2
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float32)
        model = PeftModel.from_pretrained(model, str(adapter))
        merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(out))
    sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    info = {"versions": {p: version(p) for p in ("peft", "torch", "transformers")}, "adapter_sha256": sha,
            "dtype": "float32", "warnings": [str(w.message) for w in caught]}
    (out / "peft_merge.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
