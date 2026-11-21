import pytest
import torch
from wsp.hybrid_conv import causal_conv1d_fn, causal_conv1d_update


@pytest.mark.parametrize('channels', [8, 128])
@pytest.mark.parametrize('length', [1, 3, 17])
@pytest.mark.parametrize('activation', [None, 'silu'])
def test_convolution_and_state_match_dense(channels, length, activation):
    torch.manual_seed(49)
    x = torch.randn(1, channels, length)
    w, b = torch.randn(channels, 4), torch.randn(channels)
    def ref(x, padding, count):
        y = torch.nn.functional.conv1d(x, w[:, None], b, padding=padding, groups=channels)
        y = y[:, :, :count] if padding else y[:, :, -count:]
        return torch.nn.functional.silu(y) if activation else y
    torch.testing.assert_close(causal_conv1d_fn(x,w,b,activation),ref(x,3,length),atol=2e-6,rtol=2e-6)
    state = torch.randn(1,channels,4)
    joined = torch.cat((state.clone(),x),dim=-1)
    result = causal_conv1d_update(x,state,w,b,activation)
    torch.testing.assert_close(result,ref(joined,0,length),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(state,joined[:,:,-4:],atol=0,rtol=0)
