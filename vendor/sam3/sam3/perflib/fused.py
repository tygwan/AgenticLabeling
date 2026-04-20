# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    """Fused linear + activation. Falls back to standard ops on pre-Ampere GPUs."""
    x = linear(mat1)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        return torch.nn.functional.relu(x)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        return torch.nn.functional.gelu(x)
    raise ValueError(f"Unexpected activation {activation}")
