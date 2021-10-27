import json

import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from urllib.parse import urljoin
import torch
import sys

sys.path.insert(0, '../common')
from tensor_utils import quantize_tensor




class API:
    def __init__(self, base_url):
        self.base_url = base_url
        self.encoder_model = torch.jit.load('../server/assets/effd2_encoder.ptl')
        self.encoder_model.eval()
        self.encoder_model.set_width(1.0)

    def set_width(self, width):
        self.encoder_model.set_width(width)

    def delete_results(self):
        url = urljoin(self.base_url, "map")
        res = requests.delete(url=url)
        return res

    def get_results(self):
        url = urljoin(self.base_url, "map")
        res = requests.get(url=url)
        return json.loads(res.content)

    def post_data(self, filename, json_data):
        url = urljoin(self.base_url, "data/" + filename)
        res = requests.post(url=url, json=json_data)
        return res

    def split_offload(self, image, image_id):
        url = urljoin(self.base_url, "split")
        w = image.size[0]
        h = image.size[1]

        with torch.no_grad():
            x = tfunc.to_tensor(image)
            x = x.unsqueeze(0)
            x = self.encoder_model(x) # approx 200ms

        x = quantize_tensor(x, num_bits=8)
        res = requests.post(url=url,
                            data=x.tensor.numpy().astype(np.uint8).tobytes(),
                            headers={'Content-Type': 'application/octet-stream',
                                     "image_id": image_id,
                                     "w": str(w), "h": str(h),
                                     "scale": str(float(x.scale)),
                                     "zero_point": str(float(x.zero_point))})

        return res


#################### Deprecated #####################

def offload(base_url, image, image_id):
    url = urljoin(base_url, "compute")
    w = image.size[0]
    h = image.size[1]
    # image = image.resize((640, 640))
    x = tfunc.to_tensor(image)
    x = (x * 255).numpy().astype(np.uint8)
    res = requests.post(url='http://0.0.0.0:5000/compute',
                        data=x.tobytes(),
                        headers={'Content-Type': 'application/octet-stream',
                                 "image_id": image_id,
                                 "w": str(w),
                                 "h": str(h)})

    return res
