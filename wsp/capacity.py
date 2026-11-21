"""Interactive defaults and native per-call bounds; model metadata sets total context."""
DEFAULT_CONTEXT = 8192  # Fallback only when the model window is unknown.
DEFAULT_OUTPUT = 2048
PREFILL_CHUNK = 128
NATIVE_ROWS = {'cpu': 2048, 'metal': 512}
NATIVE_MAX_THREADS = 32  # Pinned ABI 1 CPU bridge input guard.
NATIVE_BYTES = 128 * 1024**2
MAX_EXACT_INTEGER = 2**53-1  # JSON/browser integer precision, not a runtime capacity policy.


def context_limit(configured, model_context=None):
    return min(configured, model_context) if model_context else configured


def output_count(requested, prompt_tokens, context, *, allow_zero=False, maximum=None):
    minimum = 0 if allow_zero else 1
    if type(requested) is not int or not minimum <= requested <= MAX_EXACT_INTEGER:
        raise ValueError(f'Output must be a finite whole number >= {minimum}')
    if maximum is not None and requested > maximum:
        raise ValueError(f'Output ceiling exceeds the active maximum of {maximum}; change Settings and restart to raise it')
    if context is None:
        return requested
    remaining = context-prompt_tokens
    if remaining < 0 or (requested > 0 and remaining == 0):
        raise ValueError(f'Prompt/history uses {prompt_tokens} tokens; active context is {context}, leaving no response space. No history was truncated.')
    return min(requested, remaining)
