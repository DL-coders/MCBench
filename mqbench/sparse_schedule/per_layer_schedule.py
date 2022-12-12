import torch

from mqbench.sparse_schedule.base_schedule import _Scheduler
from mqbench.utils.logger import logger

class PerLayerScheduler(_Scheduler):
    def __init__(self, ratio, metric='l2'):
        super().__init__()
        self.ratio = ratio 
        assert metric in ['l2', 'mag', 'grad']
        self.metric = metric

    @property
    def sparse_modules(self):
        return (
            torch.nn.Linear, 
        )

    def get_mask(self, model: torch.nn.Module):
        all_weight = []
        return_ratio = {}
        for name, mod in model.named_modules():
            if isinstance(mod, self.sparse_modules):
                all_weight.append(mod.weight.flatten())
                prune_num = int(self.ratio * mod.weight.numel())
                if prune_num == 0:
                    rate = 0.
                else:
                    thresh = mod.weight.flatten().sort()[0][prune_num - 1]
                    rate = (mod.weight < thresh) / mod.weight.numel()
                return_ratio[name] = rate
        return return_ratio

    def update_sparse_config(self, model: torch.nn.Module, ratio_dict: dict):
        for name, mod in model.named_modules():
            if name in ratio_dict:
                mod.weight_fake_sparse.ratio = ratio_dict[name]
                logger.info(f'{name} set by {self.metric} as {ratio_dict[name]}')
        return model 
    
    def __call__(self, model):
        ratio = self.get_mask(model)
        model = self.update_sparse_config(model, ratio)
        return model

        