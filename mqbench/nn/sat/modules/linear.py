import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear
    def __init__(self, in_features, out_features, bias=True, sconfig=None, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_fake_sparse = sconfig()

    def forward(self, input):
        return F.linear(input, self.weight_fake_sparse(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, sconfig):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__

        sat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, sconfig=sconfig)
        sat_linear.weight = mod.weight
        sat_linear.bias = mod.bias
        return sat_linear