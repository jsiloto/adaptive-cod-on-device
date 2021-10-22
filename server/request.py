import os
import sys

import numpy as np
import torch

sys.path.insert(0, '../common')
from tensor_utils import dequantize_tensor, QuantizedTensor

base_path = "/workspace/resource/dataset/coco2017/val2017/"


class RequestParser:
    def __init__(self, request):
        self.w = int(request.headers['w'])
        self.h = int(request.headers['h'])
        self.image_id = int(request.headers['image_id'].split('.jpg')[0])
        data = np.fromstring(request.data, dtype=np.uint8)
        self.alpha = len(data) / (1 * 48 * 96 * 96)
        self.data = data.reshape([1, int(48 * self.alpha), 96, 96])
        self.image_path = os.path.join(base_path, request.headers['image_id'])

        self.scale = float(request.headers['scale'])
        self.zero_point = float(request.headers['zero_point'])
        x = QuantizedTensor(tensor=torch.tensor(self.data), scale=self.scale, zero_point=self.zero_point)
        self.dequantized_data = dequantize_tensor(x)

    def __str__(self):
        s = "################# ID:{}################\n".format(self.image_id)
        s += "w:{} h:{}\nScale:{}\nZero Point:{}\n{}\n".format(
            self.w, self.h,
            self.scale, self.zero_point,
            self.dequantized_data
        )
        s += "######################################"
        return s