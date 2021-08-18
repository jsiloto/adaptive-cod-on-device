import requests

import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import bitstream
from bitstream import BitStream

from functools import reduce  # Required in Python 3
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


image_path = './resource/000000000161.jpg'

image = Image.open(image_path)
x = TF.to_tensor(image)
x.unsqueeze(0)
print(x.shape)

x = x.numpy().astype(np.int8)
print(prod(x.shape))
print(len(x.tostring())/4)

#
#

#
res = requests.post(url='http://0.0.0.0:5000/png',
                    data=x.tostring(),
                    headers={'Content-Type': 'application/octet-stream'})


with open('./resource/000000000161.jpg', 'rb') as f:
    data = f.read()
res = requests.post(url='http://0.0.0.0:5000/jpg',
                    data=data,
                    headers={'Content-Type': 'application/octet-stream'})

# Build model
# Load model from checkpoint
# Load image and convert to tensor
# Run Model (or not)
# Serialize tensor
# upload tensor