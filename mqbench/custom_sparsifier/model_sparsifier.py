import copy
import operator
from collections import OrderedDict
from typing import (
    List, Dict, Any, Callable
)

import torch
from torch.fx import (
    GraphModule
)

import mqbench.nn.sat as qnnsat
from mqbench.utils.logger import logger


def swap_module(mod, sconfig, mapping):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    swapped = False
    if type(mod) in mapping:
        new_mod = mapping[type(mod)]
        swapped = True
        new_mod = new_mod.from_float(mod, sconfig)
        if swapped:
            # Preserve module's pre forward hooks. They'll be called on quantized input
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            # Preserve module's post forward hooks except _observer_forward_hook
            # After convert they'll work with quantized output
    return new_mod

class ModelSparsifier(object):

    def __init__(self, exclude_module_name)
        if exclude_module_name:
            self.exclude_module_name = exclude_module_name
        else:
            self.exclude_module_name = []
        self.mapping = {
            torch.nn.Linear: qnnsat.Linear
        }

    def prepare(self, model: GraphModule, sconfig):
        model = self._weight_sparse(model, sconfig)
        return model

    def _weight_sparse(self, model: GraphModule, sconfig):
        logger.info("Replace module to sat module.")
        self._sat_swap_modules(model, self.mapping, sconfig)
        return model

    def _sat_swap_modules(self, root: GraphModule, sparse_module_mapping, sconfig):
        root = self._convert(root, sparse_module_mapping, sconfig, inplace=True)
        return root

    def _convert(self, module, mapping=None, sconfig=None, inplace=False, scope=''):
        if mapping is None:
            raise ValueError('mapping is None')

        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            # fused modules are swapped as one unit
            new_scope = "{}.{}".format(scope, name) if scope != '' else name
            if new_scope in self.exclude_module_name:
                logger.info("Skip sparse layer: " + new_scope)
                continue
            if not type(mod) in mapping:
                self._convert(mod, mapping, sconfig, True, new_scope)
            reassign[name] = swap_module(mod, sconfig, mapping)
        for key, value in reassign.items():
            module._modules[key] = value

        return module

if __name__ == '__main__':
    from mqbench.utils.state import enable_sparse, disable_sparse
    model = torch.nn.Sequential(
        torch.nn.Linear(3,4,True),
        torch.nn.Linear(4,4,True)
    )
    sparsifier = ModelSparsifier()
    from mqbench.fake_sparsity.norm import NormFakeSparse
    sconfig = NormFakeSparse.with_args(ratio=0.2, norm=2, fixed_mask=False)
    smodel = sparsifier.prepare(model, sconfig)
    data = torch.rand(10, 3)
    enable_sparse(smodel)
    loss = smodel(data).sum()
    loss.backward()
    print(smodel)
    disable_sparse(smodel)
    print(smodel)