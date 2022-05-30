import torch

from torch import nn
from literature_models.common.efficientdet.blocks import MBConvBlockV2
from literature_models.common.efficientdet.slimmable_ops import USMBConvBlock
from literature_models.common.efficientdet.utils import get_model_params, BlockArgs, Swish, get_same_padding_conv2d, \
    round_filters


class Assine2022AEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        _, global_params = get_model_params(model_name='efficientnet-b2', override_params=None)
        # Batch norm parameters
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, global_params)  # number of output channels
        self.conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.swish = Swish()
        self.blocks = nn.ModuleList([])

        args_list = [
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=32, output_filters=16,
                      expand_ratio=1, id_skip=True, stride=[1], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=16,
                      expand_ratio=1, id_skip=True, stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=16, output_filters=24,
                      expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=24, output_filters=24,
                      expand_ratio=6, id_skip=True, stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=24, output_filters=24,
                      expand_ratio=6, id_skip=True, stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=3, input_filters=24, output_filters=48,
                      expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=3, input_filters=48, output_filters=48,
                      expand_ratio=6, id_skip=True, stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=3, input_filters=48, output_filters=48,
                      expand_ratio=6, id_skip=True, stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=4, input_filters=48, output_filters=48,
                      expand_ratio=6, id_skip=False, stride=[1], se_ratio=0.25)
        ]

        # Not Slimmable
        for bargs in args_list[0:2]:
            new_block = USMBConvBlock(bargs, global_params, slimmable_input=False,
                                    slimmable_output=False, fully_slimmable=False)
            self.blocks.append(new_block)

        # Slimmable Transition
        for bargs in args_list[2:3]:
            trans_block = USMBConvBlock(bargs, global_params, slimmable_input=False)
            self.blocks.append(trans_block)


        for bargs in args_list[3:-1]:
            new_block = USMBConvBlock(bargs, global_params)
            self.blocks.append(new_block)

        last_block = USMBConvBlock(bargs, global_params, slimmable_output=False)
        self.blocks.append(last_block)

    @torch.jit.export
    def set_mode(self, mode: int):
        alpha: float = mode / 100.0
        for m in self.modules():
            if hasattr(m, "width_mult"):
                m.width_mult = alpha
            if hasattr(m, "width_mult_in"):
                m.width_mult_in = alpha
            if hasattr(m, "width_mult_out"):
                m.width_mult_out = alpha


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.swish(x)
        for b in self.blocks:
            x = b(x)
        return x


