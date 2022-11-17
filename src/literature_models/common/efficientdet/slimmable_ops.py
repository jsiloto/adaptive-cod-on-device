import math
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor
from torch.nn import InstanceNorm2d

from literature_models.common.efficientdet.utils import get_same_padding_conv2d, Swish


def float_index(a: float, floats):
    # l = np.isclose(a, floats, rtol=0.01).nonzero()[0]
    l = torch.isclose(torch.tensor(a), floats, rtol=0.01).nonzero()[0]
    if len(l) == 1:
        return int(l[0])
    else:
        raise ValueError

class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_mult_list, ratio=1, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        num_features_max = int(round(num_features, 0))
        super(USBatchNorm2d, self).__init__(num_features_max, momentum=momentum, affine=affine, eps=eps,
                                            track_running_stats=track_running_stats)
        self.num_features_basic = num_features
        self.width_mult_list = torch.tensor(width_mult_list)
        # self.width_mult_list = width_mult_list
        # for tracking log during training
        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [int(round(num_features * width_mult)) for width_mult in width_mult_list]
             ]
        )
        self.ratio = ratio
        # self.width_mult = torch.tensor(1.0)
        self.width_mult = 1.0
        self.ignore_model_profiling = True

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        bias = self.bias
        c = int(round(self.num_features_basic * self.width_mult))
        idx = float_index(self.width_mult, self.width_mult_list)

        if idx is not None:
            for i, m in enumerate(self.bn):
                if i == idx:
                    y: Tensor = nn.functional.batch_norm(
                        input,
                        m.running_mean[:c],
                        m.running_var[:c],
                        weight[:c],
                        bias[:c],
                        self.training,
                        self.momentum,
                        self.eps)
                    return y
            raise ValueError
        else:
            y: Tensor = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
            return y


class USConv2dStaticSamePadding(nn.Conv2d):
    # TODO(jsiloto) Figure out license for this
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, groups=1, dilation=1, depthwise=False,
                 slimmable_input=True, slimmable_output=True, **kwargs):
        groups = in_channels if depthwise else groups
        super().__init__(in_channels, out_channels,
                         kernel_size, stride=stride, dilation=dilation,
                         groups=groups, bias=bias)
        self.slimmable_input = slimmable_input
        self.slimmable_output = slimmable_output
        self.width_mult_in: float = 1.0
        self.width_mult_out: float = 1.0
        self.depthwise = depthwise

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        in_channels = self.in_channels
        out_channels = self.out_channels

        if self.slimmable_input:
            in_channels = int(round(self.in_channels * self.width_mult_in))
        if self.slimmable_output:
            out_channels = int(round(self.out_channels * self.width_mult_out))

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        weight = self.weight[:out_channels, :in_channels, :, :]
        groups = min(self.groups, in_channels)

        if self.bias is not None:
            bias = self.bias
            assert bias is not None
            bias = bias[:out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(x, weight, bias,
                                 self.stride, self.padding,
                                 self.dilation, groups=groups)

        return y


class USMBConvBlock(nn.Module):
    def __init__(self, block_args, global_params,
                 fully_slimmable=True, slimmable_input=True, slimmable_output=True):
        super().__init__()
        self.fully_slimmable = fully_slimmable
        self.expand_ratio = block_args.expand_ratio
        self._block_args = block_args
        self.stride = block_args.stride
        self.se_ratio = block_args.se_ratio
        self.input_filters = block_args.input_filters
        self.output_filters = block_args.output_filters
        self.kernel_size = block_args.kernel_size
        self.width_mult_list = [0.25, 0.5, 0.75, 1.00]
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        USConv2d = partial(USConv2dStaticSamePadding, image_size=global_params.image_size)
        if fully_slimmable:
            Conv2d = USConv2d

        # Expansion phase
        inp = block_args.input_filters  # number of input channels
        oup = block_args.input_filters * block_args.expand_ratio  # number of output channels
        k = self.kernel_size
        s = self.stride
        if self.expand_ratio != 1:
            self._expand_conv = USConv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False,
                                         slimmable_input=slimmable_input, slimmable_output=fully_slimmable)
            if fully_slimmable:
                self._bn0 = USBatchNorm2d(num_features=oup, width_mult_list=self.width_mult_list,
                                          momentum=self._bn_mom, eps=self._bn_eps)
            else:
                self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        else:
            self._expand_conv = nn.Identity()
            self._bn0 = nn.Identity()

        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup, depthwise=True,
            kernel_size=k, stride=s, bias=False)

        if fully_slimmable:
            self._bn1 = USBatchNorm2d(num_features=oup, width_mult_list=self.width_mult_list,
                                      momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn1 = nn.BatchNorm2d(num_features=oup,
                                       momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self.input_filters * self.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self.output_filters
        self._project_conv = USConv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False,
                                      slimmable_output=slimmable_output, slimmable_input=fully_slimmable)

        if slimmable_output:
            self._bn2 = USBatchNorm2d(num_features=final_oup, width_mult_list=self.width_mult_list,
                                      momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn2 = nn.BatchNorm2d(num_features=final_oup,
                                       momentum=self._bn_mom, eps=self._bn_eps)

        self._swish = Swish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
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

        # hack for int, list comparison on pytorch
        if self.id_skip and self.stride == 1 and x.shape == inputs.shape \
                and not isinstance(self.stride, list):
            x = x + inputs  # skip connection
        return x
