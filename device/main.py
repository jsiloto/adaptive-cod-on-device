import operator
import os
import random
from functools import reduce  # Required in Python 3

import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from PIL import Image


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


# image_path = './resource/000000000161.jpg'
# image_path = "./resource/000000112378.jpg"
# image_path = "./resource/000000000109.jpg"
image_path = './resource/img_test.png'


for i in range(10):

    image_id = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
    image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
    print(image_path)
    # image_path = "/workspace/resource/dataset/coco2017/val2017/000000534664.jpg"

    image = Image.open(image_path)
    w = image.size[0]
    h = image.size[1]


    print(image.size)
    image = image.resize((640, 640))
    x = tfunc.to_tensor(image)
    x = (x*255).numpy().astype(np.uint8)
    res = requests.post(url='http://0.0.0.0:5000/compute',
                        data=x.tobytes(),
                        headers={'Content-Type': 'application/octet-stream',
                                 "image_id": image_id,
                                 "w": str(w),
                                 "h": str(h)})


# Build model
# Load model from checkpoint
# Load image and convert to tensor
# Run Model (or not)
# Serialize tensor
# upload tensor