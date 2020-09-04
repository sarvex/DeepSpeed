import argparse
import torch

import time
import numpy as np
import pytest
import copy

from deepspeed.ops.adam import DeepSpeedCPUAdam

def check_equal(first, second, atol=1e-2, verbose=False):
    if verbose:
        print(first)
        print(second)
    x = first.detach().numpy()
    y = second.detach().numpy()
    if verbose:
        print("x = {}".format(x.flatten()))
        print("y = {}".format(y.flatten()))
        print('-' * 80)
    np.testing.assert_allclose(x, y, err_msg="param-update dismatch!", atol=atol)

@pytest.mark.parametrize('model_size',
                         [
                             (1048576),
                         ]) # yapf: disable
def test_adam_opt(model_size):
    device = 'cpu'
    rng_state = torch.get_rng_state()
    param = torch.nn.Parameter(torch.randn(model_size, device=device))
    torch.set_rng_state(rng_state)
    param1 = torch.nn.Parameter(torch.randn(model_size, device=device))

    optimizer1 = torch.optim.Adam([param1])
    optimizer = DeepSpeedCPUAdam([param])

    for i in range(10):
        rng_state = torch.get_rng_state()
        param.grad=torch.randn(model_size, device=device)
        torch.set_rng_state(rng_state)
        param1.grad=torch.randn(model_size, device=device)

        optimizer.step()
        optimizer1.step()

    check_equal(param, param1, atol = 1e-2, verbose=True)
