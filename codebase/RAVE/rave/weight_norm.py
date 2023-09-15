r"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim, Tensor
from typing import Any, TypeVar, Tuple, Optional, List
from torch.nn import Module

__all__ = [
    'weight_norm',
    'remove_weight_norm'
    ]

T_module = TypeVar('T_module', bound=Module)

def weight_norm(module: T_module) -> T_module:

    # fn = WeightNorm(name, dim)
    def fn(module: Module, inputs: Tuple[Tensor]) -> None:
        module.weight = _weight_norm(module.weight_v, module.weight_g, 0)

    fn.is_weight_norm = True

    weight = getattr(module, 'weight')
    if isinstance(weight, UninitializedParameter):
        raise ValueError(
            'The module passed to `weight_norm` can\'t have uninitialized parameters. '
            'Make sure to run the dummy forward before applying weight normalization')
    # remove w from parameter list
    del module._parameters['weight']

    # add g and v as new parameters and express w as g/||v|| * v
    module.register_parameter('weight_g', Parameter(norm_except_dim(weight, 2, 0).data))
    module.register_parameter('weight_v', Parameter(weight.data))
    fn(module, None)

    # recompute weight before every forward()
    module.register_forward_pre_hook(fn)

    return module

def remove_weight_norm(module: T_module) -> T_module:
    for k, hook in module._forward_pre_hooks.items():
        if hasattr(hook, 'is_weight_norm') and hook.is_weight_norm:
            weight = _weight_norm(module.weight_v, module.weight_g, 0)
            delattr(module, 'weight')
            del module._parameters['weight_g']
            del module._parameters['weight_v']
            module.weight = Parameter(weight.data)

            del module._forward_pre_hooks[k]
            return module