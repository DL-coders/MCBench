import torch
from torch.nn import Module


from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Tuple


class _PartialWrapper(object):
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # call each arg in callable_args and add them partial, then run with keywords
        # skip if arg_name in keywords so its possible to overwrite
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, **{arg_name: self.callable_args[arg_name]()}}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

class FakeSparseBase(ABC, Module):
    fake_sparse_enabled: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer('fake_sparse_enabled', torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_mask(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_sparse(self, enabled: bool = True) -> None:
        self.fake_sparse_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_sparse(self):
        self.enable_fake_sparse(False)

    @classmethod
    def with_args(cls, **kwargs):
        fake_sparse_constructor = _with_args(cls, **kwargs)
        return fake_sparse_constructor
