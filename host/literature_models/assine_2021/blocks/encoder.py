from typing import Tuple

import torch
from torch import nn
from torch.nn import InstanceNorm2d

from models.convert import replace_swish
from models.efficientdet.custom.blocks import MBConvBlockV2
from models.efficientdet.custom.ensemble import Ensemble
from models.efficientdet.efficientnet.utils import efficientnet_params, efficientnet, round_repeats
from models.efficientdet.model import EfficientNet
from models.mimic.bottleneck import get_channel_number
from models.slimmable.slimmable_ops import USMBConvBlock, USConv2d, USInstanceNorm2d

import models.slimmable.slimmable_ops as slim
import models.slimmable.qat_ops as slimqat
import models.slimmable.quantized_ops as slimquant

def build_encoder(original: EfficientNet, config):
    name = config['encoder']
    builder = {
        'vanilla': vanilla_encoder,
        'scaled-vanilla': scaled_vanilla_encoder,
        'minimal': minimal_encoder,
        'vanilla-quant': quantized_vanilla,
        'minimal-quant': minimal_quantized_encoder,
    }
    if config['configurability']['mode'] == 'ensemble':
        return Ensemble(builder[name], original, config)
    else:
        return builder[name](original, config)


def vanilla_encoder(original: EfficientNet, config):
    return VanillaEncoder(original, config)

def scaled_vanilla_encoder(original: EfficientNet, config):
    return ScaledVanillaEncoder(original, config)

def minimal_encoder(original: EfficientNet, config):
    bottleneck_channel = config['params']['bottleneck']['bottleneck_channel']
    encoder = MinimalEncoder(bottleneck_channel, quantized=False)
    return encoder


def quantize_encoder(encoderfp32):
    mapping = {
        slim.USConv2d: slimqat.USConv2d,
        slim.USConv2dStaticSamePadding: slimqat.USConv2dStaticSamePadding
    }

    qconfig = torch.quantization.get_default_qconfig('qnnpack')
    qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'
    replace_swish(encoderfp32, nn.Hardswish())
    encoderfp32.to('cpu')
    encoderfp32.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    encoderfp32_prepared = torch.quantization.prepare_qat(encoderfp32, inplace=True, mapping=mapping)
    encoderfp32_prepared.to('cpu')
    return encoderfp32_prepared

def quantized_vanilla(original: EfficientNet, config):
    encoder = VanillaEncoder(original, config, quantized=True)
    encoderfp32_prepared = quantize_encoder(encoder)
    return encoderfp32_prepared

def minimal_quantized_encoder(original: EfficientNet, config):
    bottleneck_channel = config['params']['bottleneck']['bottleneck_channel']
    encoderfp32 = MinimalEncoder(bottleneck_channel, quantized=True)
    encoderfp32_prepared = quantize_encoder(encoderfp32)
    return encoderfp32_prepared


class VanillaEncoder(nn.Module):
    def __init__(self, original: EfficientNet, config, quantized=False):
        super(VanillaEncoder, self).__init__()
        global_params = original.model._global_params
        self.conv_stem = original.model._conv_stem
        self.bn0 = original.model._bn0
        self.blocks = nn.ModuleList([])
        self.quantized = quantized
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.bottleneck_channel = config['params']['bottleneck']['bottleneck_channel']
        self.original_channels = get_channel_number(config)


        num_bottlenecks = 0
        slimmable_input = False
        split_location = config['params']['split_location']

        print("Building Custom EfficientNetEncoder")
        assert split_location <= 3
        for idx, block in enumerate(original.model._blocks):
            if block._depthwise_conv.stride == [2, 2]:
                num_bottlenecks += 1

            if num_bottlenecks == 0:
                # print("Append Original Blocks", idx)
                new_block = USMBConvBlock(block._block_args, global_params, config,
                                          slimmable_input=False,
                                          slimmable_output=False,
                                          fully_slimmable=False,
                                          quantized=self.quantized)
                self.blocks.append(new_block)

            elif 0 < num_bottlenecks < split_location:
                # print("Append Slimmable Blocks", idx)
                new_block = USMBConvBlock(block._block_args, global_params, config,
                                          slimmable_input=slimmable_input,
                                          slimmable_output=True,
                                          fully_slimmable=True,
                                          quantized=self.quantized)
                self.blocks.append(new_block)
                slimmable_input = True


        w, d, s, p = efficientnet_params("efficientnet-b{}".format(original.compound_coef))
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d,
                                                  dropout_rate=p, image_size=s)

        block_args = blocks_args[4]

        block_args_encoder = block_args._replace(
            input_filters=self.original_channels,
            output_filters=self.bottleneck_channel,
            stride=1,
            id_skip=False,
            num_repeat=round_repeats(block_args.num_repeat, global_params)
        )
        self.last_block = USMBConvBlock(block_args_encoder, global_params, config=config,
                                slimmable_input=True, slimmable_output=True,
                                fully_slimmable=True)


    def forward(self, x):
        if self.quantized:
            x = self.quant(x)
        x = self.conv_stem(x)
        x = self.bn0(x)
        for b in self.blocks:
            x = b(x)

        x = self.last_block(x)

        if self.quantized:
            x = self.dequant(x)
        return x

    def set_config(self, config):
        bw_width, cp_width = config[0], config[1]
        for m in self.modules():
            if hasattr(m, "width_mult"):
                m.width_mult = cp_width
            if hasattr(m, "width_mult_in"):
                m.width_mult_in = cp_width
            if hasattr(m, "width_mult_out"):
                m.width_mult_out = cp_width

        self.last_block._project_conv.width_mult_out = bw_width
        self.last_block._bn2.width_mult = bw_width

class MinimalEncoder(nn.Module):

    def __init__(self, bottleneck_channels=24, quantized=False):
        super().__init__()
        self.quantized = quantized
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv_stem = slim.USConv2d(3, 32, stride=2, kernel_size=3, padding=1,
                                     slimmable_input=False)
        # self.conv_stem = nn.Conv2d(3, 16, kernel_size=5, stride=2, bias=False, padding=2)
        self.norm0 = nn.InstanceNorm2d(32)
        self.activation = nn.Hardswish(inplace=False)
        self.usconv1 = slim.USConv2d(32, 32, stride=2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(32)
        self.usconv2 = slim.USConv2d(32, bottleneck_channels, stride=3, kernel_size=3, padding=0, bias=False)
        self.norm2 = nn.InstanceNorm2d(bottleneck_channels)

    def forward(self, x):
        if self.quantized:
            x = self.quant(x)
        x = self.conv_stem(x)

        x = self.norm0(x)
        x = self.activation(x)

        x = self.usconv1(x)

        x = self.norm1(x)
        x = self.activation(x)

        x = self.usconv2(x)

        x = self.norm2(x)
        x = self.activation(x)
        # print(x[0][0][0][1:5], x[0][0][-1][-5:])
        if self.quantized:
            x = self.dequant(x)
        return x

    def set_config(self, config):
        bw_width, cp_width = config[0], config[1]
        for m in self.modules():
            if hasattr(m, "width_mult"):
                m.width_mult = cp_width
            if hasattr(m, "width_mult_in"):
                m.width_mult_in = cp_width
            if hasattr(m, "width_mult_out"):
                m.width_mult_out = cp_width

        self.usconv2.width_mult_out = bw_width


class ScaledVanillaEncoder(nn.Module):
    def __init__(self, original: EfficientNet, config):
        super(ScaledVanillaEncoder, self).__init__()
        global_params = original.model._global_params
        bw_width, cp_width = config['configurability']['config']
        last_block_output = 24
        last_block_output = int(last_block_output * cp_width)
        # Batch norm parameters
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        self.conv_stem = nn.Conv2d(3, last_block_output, kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=last_block_output, momentum=bn_mom, eps=bn_eps)
        self.blocks = nn.ModuleList([])
        self.bottleneck_channel = config['params']['bottleneck']['bottleneck_channel']
        self.original_channels = get_channel_number(config)

        num_bottlenecks = 0
        num_repeat = 0

        split_location = config['params']['split_location']

        print("Building Custom EfficientNetEncoder")
        assert split_location <= 3
        for idx, block in enumerate(original.model._blocks):
            if block._depthwise_conv.stride == [2, 2]:
                num_bottlenecks += 1
                num_repeat = 0

            if num_bottlenecks == 0:
                # print("Append Original Blocks", idx)
                bargs = block._block_args
                bargs = bargs._replace(input_filters=last_block_output,
                               output_filters=int(bargs.output_filters*cp_width))
                new_block = MBConvBlockV2(bargs, global_params)
                last_block_output = bargs.output_filters
                self.blocks.append(new_block)
                print(bargs)
                num_repeat +=1


            elif 0 < num_bottlenecks < split_location and num_repeat < 2:
                # print("Append Original Blocks", idx)
                bargs = block._block_args
                bargs = bargs._replace(input_filters=last_block_output,
                               output_filters=int(bargs.output_filters*cp_width))
                new_block = MBConvBlockV2(bargs, global_params)
                last_block_output = bargs.output_filters
                self.blocks.append(new_block)
                num_repeat += 1
                print(bargs)


        bargs = bargs._replace(
            input_filters=last_block_output,
            output_filters=int(self.bottleneck_channel*bw_width),
            stride=1,
            id_skip=False,
        )
        self.last_block = MBConvBlockV2(bargs, global_params)
        print(bargs)


    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        for b in self.blocks:
            x = b(x)

        x = self.last_block(x)
        # torch.clip(x, min=-2.0, max=2.0, out=x)
        return x

    def set_config(self, config):
        pass
