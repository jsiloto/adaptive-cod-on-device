import os
import sys

import numpy as np
import torch

sys.path.insert(0, '../common')
from tensor_utils import dequantize_tensor, QuantizedTensor

base_path = "/work/resource/dataset/coco2017/val2017/"


class RequestParser:
    def __init__(self, request):
        self.w = int(request.headers['w'])
        self.h = int(request.headers['h'])
        self.ct = int(request.headers['ct'])
        self.wt = int(request.headers['wt'])
        self.ht = int(request.headers['ht'])
        self.alpha = float(request.headers['width'])
        try:
            self.image_id = int(request.headers['image_id'].split('.jpg')[0])
        except ValueError:
            self.image_id = 0
        data = np.fromstring(request.data, dtype=np.uint8)
        self.data = data.reshape([1, self.ct, self.wt, self.ht])
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