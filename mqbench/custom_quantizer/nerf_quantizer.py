import operator
from typing import List
from collections import OrderedDict

import torch
from torch.fx import GraphModule

import mqbench.nn.qat as qnnqat
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.NeRF)
class NeRFModelQuantizer(ModelQuantizer):
    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.nerf_acti_modules = (
            torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU,
            torch.nn.ReLU
        )
        self.nerf_acti_functions = (torch.nn.functional.relu, torch.nn.functional.relu6)


    def _insert_fake_quantize_for_act_quant(self, model: GraphModule, qconfig):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        for node in node_to_quantize_output:
            fake_quantizer = qconfig.activation()
            if node in self.weight_nodes:
                quantizer_name = node.name + '_weight_fake_quant'
                fake_quantizer = qconfig.weight()
            else:
                quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model

    def _find_act_quants(self, model: GraphModule) -> set:
        nodes = list(model.graph.nodes)
        self.weight_nodes = []
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        for node in nodes:
            if (node.op == "call_module" and isinstance(modules[node.target], self.nerf_acti_modules)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                        node.target in self.nerf_acti_functions):
                node_need_to_quantize_output.append(node)
            elif (node.op == 'call_function' and node.target == torch.nn.functional.grid_sample):
                node_need_to_quantize_output.append(node.args[0])
                self.weight_nodes.append(node.args[0])
        return node_need_to_quantize_output


