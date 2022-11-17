import collections

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, tensor

from literature_models.common.efficientdet.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    BlockArgs,
)
from literature_models.common.efficientdet.utils_extra import Conv2dStaticSamePadding


class MBConvBlockV2(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self.expand_ratio = block_args.expand_ratio
        self._block_args = block_args
        self.stride = self._block_args.stride
        self.se_ratio = self._block_args.se_ratio
        self.input_filters = self._block_args.input_filters
        self.output_filters = self._block_args.output_filters
        self.kernel_size = self._block_args.kernel_size
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self.input_filters  # number of input channels
        oup = self.input_filters * self.expand_ratio  # number of output channels
        k = self.kernel_size
        s = self.stride

        self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, bias=False,
                                   kernel_size=k, stride=s)
        self._bn0 = nn.InstanceNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self.input_filters * self.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.InstanceNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs

        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.input_filters, self.output_filters
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            x = x + inputs  # skip connection
        return x


class MBConvTransposeBlockV2(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, input_filters, output_filters, kernel_size=3, stride=2, upsample=False, id_skip=False):
        super().__init__()
        self.stride = stride

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.upsample = upsample

        self._bn_mom = 0.99
        self._bn_eps = 1e-3
        self.has_se = True
        self.se_ratio = 0.25
        self.expand_ratio = 6
        self.id_skip = id_skip
        # Get static or dynamic convolution depending on image size
        Conv2d = Conv2dStaticSamePadding

        # Expansion phase
        inp = self.input_filters  # number of input channels
        oup = self.input_filters * self.expand_ratio  # number of output channels
        k = self.kernel_size
        s = self.stride

        if self.upsample:
            self._expand_conv = nn.ConvTranspose2d(in_channels=inp,
                                                   out_channels=oup, bias=False,
                                                   kernel_size=k, stride=s)
        else:
            self._expand_conv = Conv2d(in_channels=inp,
                                       out_channels=oup, bias=False,
                                       kernel_size=k, stride=s)
        self._bn0 = nn.InstanceNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self.input_filters * self.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.InstanceNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs

        x = self._expand_conv(inputs)
        if self.upsample:
            x = x[:, :, :-1, :-1]
        x = self._bn0(x)
        x = self._swish(x)
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)
        # Skip connection and drop connect
        input_filters, output_filters = self.input_filters, self.output_filters
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            x = x + inputs  # skip connection
        return x