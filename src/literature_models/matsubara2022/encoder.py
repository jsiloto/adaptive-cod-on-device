# From https://github.com/yoshitomo-matsubara/supervised-compression/blob/main/custom/backbone.py#L132-L183
import os

import torch
from torch import nn
from compressai.layers import GDN1

class MatsubaraEntropicEncoder(nn.Module):
    def __init__(self, num_enc_channels=16, num_target_channels=256, analysis_config=None):
        super(MatsubaraEntropicEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=[800, 800], mode='nearest')
        x = self.encoder(x)
        return x

    @torch.jit.export
    def set_mode(self, mode: int):
        pass

# if __name__ == '__main__':
#     encoder = MatsubaraEntropicEncoder()
#     scripted = torch.jit.script(encoder)
#     out_dir = "./out"
#     scripted._save_for_lite_interpreter(
#         os.path.join(out_dir, "{}_encoder.ptl".format("matsubara")))
