from torch import nn
from literature_models.common.efficientdet.blocks import MBConvBlockV2
from literature_models.common.efficientdet.utils import get_model_params, BlockArgs


class Assine2022AEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        last_block_output = 24
        cp_width=0.25
        last_block_output = int(last_block_output * cp_width)
        _, global_params = get_model_params(model_name='efficientnet-b2', override_params=None)
        # Batch norm parameters
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        self.conv_stem = nn.Conv2d(3, last_block_output, kernel_size=3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=last_block_output)
        self.blocks = nn.ModuleList([])

        args_list = [
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=6,
                      output_filters=4, expand_ratio=1, id_skip=True,
                      stride=[1], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=4,
                      output_filters=4, expand_ratio=1, id_skip=True,
                      stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=4,
                      output_filters=6, expand_ratio=6, id_skip=True,
                      stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=6,
                      output_filters=6, expand_ratio=6, id_skip=True,
                      stride=1, se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=6,
                      output_filters=6, expand_ratio=6, id_skip=False,
                      stride=1, se_ratio=0.25)
        ]

        for bargs in args_list:
            new_block = MBConvBlockV2(bargs, global_params)
            self.blocks.append(new_block)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        for b in self.blocks:
            x = b(x)

        return x
