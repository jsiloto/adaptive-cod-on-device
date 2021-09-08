import operator
import os
import random
from functools import reduce  # Required in Python 3
import time

import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from PIL import Image
from api import delete_results, get_results, compute_image

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def get_random_image():
    image_id = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
    image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
    print(image_path)
    image = Image.open(image_path)
    return image, image_id


if __name__ == '__main__':
    base_url = 'http://0.0.0.0:5000/'
    delete_results(base_url)
    num_images = 5
    for i in range(num_images):
        start = time.time()
        print("Sending image {} of {}".format(i, num_images))
        image, image_id = get_random_image()

        compute_image(base_url, image, image_id)

        end = time.time()
        elapsed = (end-start)
        print("Sample Processing Time: {}".format(elapsed))

    result = get_results(base_url)
    print(result.content)

# Build model
# Load model from checkpoint
# Load image and convert to tensor
# Run Model (or not)
# Serialize tensor
# upload tensor