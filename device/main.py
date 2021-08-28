import requests
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tfunc
from torchvision import transforms

from functools import reduce  # Required in Python 3
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


# image_path = './resource/000000000161.jpg'
# image_path = "./resource/000000112378.jpg"
# image_path = "./resource/000000000109.jpg"
image_path = './resource/img_test.png'

import os, random
image_path = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_path)
print(image_path)


image = Image.open(image_path)
image = image.resize((640, 640))
print(image)
x = tfunc.to_tensor(image)
x = (x*255).numpy().astype(np.uint8)
print(x)
print(prod(x.shape))
print(len(x.tobytes()))


#
res = requests.post(url='http://0.0.0.0:5000/compute',
                    data=x.tobytes(),
                    headers={'Content-Type': 'application/octet-stream'})


# Build model
# Load model from checkpoint
# Load image and convert to tensor
# Run Model (or not)
# Serialize tensor
# upload tensor