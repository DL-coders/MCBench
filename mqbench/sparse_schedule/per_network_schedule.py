import torch

from mqbench.sparse_schedule.base_schedule import _Scheduler
from mqbench.utils.logger import logger

class PerNetWorkScheduler(_Scheduler):
    def __init__(self, ratio, metric='mag'):
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
                return_ratio[name] = (mod, mod.weight)
        all_weight = torch.cat(all_weight)
        if self.metric == 'mag':
            all_metric = all_weight.abs()
        else:
            raise NotImplementedError(f'{self.metric}')
        prune_num = int(self.ratio * all_metric.numel())
        if prune_num <= 0:
            threshold = all_metric.min() - 1
        elif prune_num >= all_metric.numel():
            threshold = all_metric.max() + 1
        else:
            threshold = all_metric.sort()[0][prune_num - 1]
        ratio_dict = {}
        for name in return_ratio:
            mod, weight = return_ratio[name]
            if self.metric == 'mag':
                ratio = (weight.abs() < threshold).sum() / weight.numel()
            ratio_dict[name] = ratio
        return ratio_dict

    def update_sparse_config(self, model: torch.nn.Module, ratio_dict: dict):
        for name, mod in model.named_modules():
            if name in ratio_dict:
                mod.weight_fake_sparse.ratio = ratio_dict[name]
                mod.weight_fake_sparse.mask = None
                logger.info(f'{name} set by {self.metric} as {ratio_dict[name]}')
        return model

    def __call__(self, model):
        ratio = self.get_mask(model)
        model = self.update_sparse_config(model, ratio)
        return model