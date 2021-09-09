import operator
import os
import random
from functools import reduce  # Required in Python 3
import time

import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from PIL import Image
from api import delete_results, get_results, offload, split_offload

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def random_image_list(num_images):
    image_list = []
    for i in range(num_images):
        image_id = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
        # image_id = "000000130826.jpg"

        image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
        image = Image.open(image_path)
        image_list.append((image, image_id))
    return image_list



def compute_image_list(image_list, compute_func):
    for i, (image, image_id) in enumerate(image_list):
        start = time.time()

        image, image_id = image_list[i]
        compute_func(base_url, image, image_id)

        end = time.time()
        elapsed = (end-start)
        print("[{}], Sample Processing Time: {}".format(image.size, elapsed))



if __name__ == '__main__':
    base_url = 'http://0.0.0.0:5000/'
    # delete_results(base_url)
    num_images = 1
    image_list = random_image_list(num_images)
    print("###############################")
    print("Computing using Full Offloading")
    compute_image_list(image_list, compute_func=offload)
    # result = get_results(base_url)

    # delete_results(base_url)
    print("###############################")
    print("Computing using Split Computing")
    compute_image_list(image_list, compute_func=split_offload)

    # result = get_results(base_url)
    # print(result.content)

# Build model
# Load model from checkpoint
# Load image and convert to tensor
# Run Model (or not)
# Serialize tensor
# upload tensor