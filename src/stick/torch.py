"""Enhances stick.flatten to handle common torch types."""
from stick.flat_utils import flatten, declare_processor
import torch
from torch import nn


@declare_processor(torch.Tensor)
def process_tensor(tensor, key, dst):
    if tensor.flatten().shape == (1,):
        dst[key] = tensor.flatten().item()
    else:
        dst[f"{key}.mean()"] = torch.mean(tensor, dtype=torch.float32).item()
        try:
            dst[f"{key}.std()"] = torch.std(tensor).item()
        except RuntimeError:
            pass


@declare_processor(nn.Module)
def process_module(module, key, dst):
    for name, param in module.named_parameters():
        flatten(param, f"{key}.{name}", dst)
        if param.grad is not None:
            flatten(param.grad, f"{key}.{name}.grad", dst)


@declare_processor(torch.optim.Optimizer)
def process_optimizer(optimizer, key, dst):
    flatten(optimizer.state_dict(), key, dst)
