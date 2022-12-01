import torch
import math
from .sparsity_base import FakeSparseBase, _with_args


class MagFakeSparse(FakeSparseBase):
    def __init__(self, ratio=0.1, fixed_mask=False):
        super().__init__()
        self.ratio = ratio
        self.fixed_mask = fixed_mask
        self.mask = None

    def __repr__(self):
        return f'MagFakeSparse(ratio={self.ratio}, fixed_mask={self.fixed_mask}, fake_quant_enabled={self.fake_sparse_enabled.item() == 1}'

    def forward(self, x):
        if self.fake_sparse_enabled[0] == 1:
            mask = self.calculate_mask(x)
            return x * mask
        else:
            return x

    def generate_mask_by_norm(self, x):
        if self.ratio >= 1:
            return torch.zeros_like(x)
        elif self.ratio <= 0:
            return torch.ones_like(x)
        mag = x.abs()
        sorted_norm = mag.flatten().sort()[0]
        masked_index = math.floor(len(sorted_norm) * self.ratio)
        if (masked_index == 0):
            masked_index += 1
        mask = mag >= (sorted_norm[masked_index])
        return mask * 1.0 

    @torch.no_grad()
    def calculate_mask(self, x):
        if self.mask is None:
            self.mask = self.generate_mask_by_norm(x)
        else:
            if self.fixed_mask is False:
                self.mask = self.generate_mask_by_norm(x)
        return self.mask
                
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
