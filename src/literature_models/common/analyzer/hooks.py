import numpy as np
from literature_models.common.efficientdet.slimmable_ops import USBatchNorm2d, USConv2dStaticSamePadding

def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)

    module.__flops__ *= module.width_mult
    module.slimmable_params = module.__params__ * module.width_mult

def usconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one

    input = input[0]
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)

    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels

    if conv_module.slimmable_input:
        in_channels = int(round(conv_module.in_channels * conv_module.width_mult_in, 0))
    if conv_module.slimmable_output:
        out_channels = int(round(conv_module.out_channels * conv_module.width_mult_out, 0))

    groups = conv_module.groups
    groups = min(conv_module.groups, in_channels)

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)

    def empty_flops_counter_hook(module, input, output):
        module.__flops__ = 0

    # conv_module.conv.__flops_handle__.remove()
    if hasattr(conv_module, 'conv'):
        conv_module.conv.register_forward_hook(empty_flops_counter_hook)

    # Hack number of parameters to be slimmable
    conv_module.slimmable_params = conv_per_position_flops


# BoilerPlate
custom_module_mapping = {
    USConv2dStaticSamePadding: usconv_flops_counter_hook,
    USBatchNorm2d: bn_flops_counter_hook,
}
