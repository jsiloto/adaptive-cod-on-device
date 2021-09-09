import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from urllib.parse import urljoin
import torch
import sys
sys.path.insert(0, '../common')
from tensor_utils import quantize_tensor

encoder_model = torch.jit.load('../server/effd2_encoder.ptl')
encoder_model.eval()


def delete_results(base_url):
    url = urljoin(base_url, "results")
    res = requests.delete(url=url)
    return res

def get_results(base_url):
    url = urljoin(base_url, "results")
    res = requests.get(url=url)
    return res


def offload(base_url, image, image_id):
    url = urljoin(base_url, "compute")
    w = image.size[0]
    h = image.size[1]
    image = image.resize((640, 640))
    x = tfunc.to_tensor(image)
    x = (x * 255).numpy().astype(np.uint8)
    res = requests.post(url='http://0.0.0.0:5000/compute',
                        data=x.tobytes(),
                        headers={'Content-Type': 'application/octet-stream',
                                 "image_id": image_id,
                                 "w": str(w),
                                 "h": str(h)})

    return res


def split_offload(base_url, image, image_id):
    url = urljoin(base_url, "compute")
    w = image.size[0]
    h = image.size[1]
    image = image.resize((640, 640))
    x = tfunc.to_tensor(image)
    x = x.unsqueeze(0)
    with torch.no_grad():
        x = encoder_model(x)

    x = quantize_tensor(x, num_bits=8)


    # x = x.numpy().astype(np.uint8)

    res = requests.post(url='http://0.0.0.0:5000/split',
                        data=x.tensor.numpy().astype(np.uint8).tobytes(),
                        headers={'Content-Type': 'application/octet-stream',
                                 "image_id": image_id,
                                 "w": str(w),
                                 "h": str(h),
                                 "scale": str(float(x.scale)),
                                 "zero_point": str(float(x.zero_point))})

    return res