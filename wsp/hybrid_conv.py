"""Bounded depthwise convolution for the pinned Qwen3.5 text forward.

PyTorch's macOS grouped conv1d dispatch costs hundreds of milliseconds for these
tiny four-tap filters. Sum the same cross-correlation explicitly; weights, cache
updates and activation remain FP32 and independently editable. No global patch.
"""
import inspect
import types

import torch
from transformers.activations import ACT2FN


def _convolve(x, weight, bias, activation, count):
    if x.ndim != 3 or weight.ndim != 2 or x.shape[1] != weight.shape[0] or not 1 <= weight.shape[1] <= 16:
        raise ValueError('Unsupported bounded depthwise convolution layout')
    width = weight.shape[1]
    start = x.shape[-1] - count - width + 1
    if start < 0:
        raise ValueError('Insufficient convolution history')
    out = x[:, :, start:start+count] * weight[None, :, 0, None]
    for tap in range(1, width):
        out.add_(x[:, :, start+tap:start+tap+count] * weight[None, :, tap, None])
    if bias is not None:
        out.add_(bias[None, :, None])
    return ACT2FN[activation](out) if activation is not None else out


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    padded = torch.nn.functional.pad(hidden_states.to(weight.dtype), (weight.shape[-1]-1, 0))
    return _convolve(padded, weight, bias, activation, hidden_states.shape[-1]).to(hidden_states.dtype)


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    joined = torch.cat((conv_state, hidden_states), dim=-1).to(weight.dtype)
    conv_state.copy_(joined[:, :, -conv_state.shape[-1]:])
    return _convolve(joined, weight, bias, activation, hidden_states.shape[-1]).to(hidden_states.dtype)


def bind(module):
    """Reuse upstream forward verbatim with two instance-local kernel bindings.

    The runner pins Transformers5.22.2. Unwrap only the accelerate-device wrapper
    (this runner has no accelerate dispatch); nn.Module hooks still surround this
    bound forward. Dense reference models retain the original upstream functions.
    """
    original = inspect.unwrap(type(module).forward)
    required = {'causal_conv1d_fn', 'causal_conv1d_update'}
    if original.__module__ != 'transformers.models.qwen3_5.modeling_qwen3_5' or not required <= set(original.__code__.co_names):
        raise RuntimeError('Pinned hybrid convolution binding changed; refusing adaptation')
    scope = dict(original.__globals__, causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=causal_conv1d_update)
    forward = types.FunctionType(original.__code__, scope, original.__name__, original.__defaults__, original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    module.forward = types.MethodType(forward, module)
