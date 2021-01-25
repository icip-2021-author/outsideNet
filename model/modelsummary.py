from os import linesep
import collections
import torch
import torch.nn as nn
from utils import async_copy_to


def get_model_summary(model, input_tensors, item_length=26, verbose=False):
    """Generates a summary of the complete model.

    Parameters
    ----------
    model : torch model
        model to evaluate
    *input_tensors : torch tensor
        data to pass through the model.
    item_length : type
        length of output items
    verbose : bool
    
    Returns
    -------
    str
        Model information as formatted string.

    """
    summary = []

    module_details = collections.namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size())))
                    * torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size())))
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                module_details(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    model.apply(add_hooks)
    space_len = item_length
    feed_dict = dict()
    feed_dict['img_data'] = input_tensors
    feed_dict['seg_label'] = torch.rand(1, 1200, 600)
    feed_dict = async_copy_to(feed_dict, 0)
    seg_size = (input_tensors.shape[2], input_tensors.shape[3])
    model(feed_dict, seg_size=seg_size)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
            + linesep + '-' * space_len * 5 + linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + linesep + '-' * space_len * 5 + linesep

    details += linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + linesep + '-' * space_len * 5 + linesep
    details += "Total Multiply Adds (only Convolution and Linear Layers): {:,} GFLOPs".format(flops_sum / (1024**3)) \
        + linesep + '-' * space_len * 5 + linesep
    details += "Number of Layers" + linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


   
