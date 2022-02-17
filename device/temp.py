import os
import random
import time

import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from urllib.parse import urljoin
import torch
import sys
from PIL import Image

sys.path.insert(0, '../common')
from tensor_utils import quantize_tensor

def random_image_list(num_images):
    image_list = []
    for i in range(num_images):
        image_id = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
        image_id = "000000133244.jpg"

        image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
        image = Image.open(image_path)
        image_list.append((image, image_id))
    return image_list



encoder_model = torch.jit.load('../server/assets/effd2_encoder.ptl')
encoder_model.eval()
encoder_model.set_width(0.25)

image_list = random_image_list(20)
with torch.no_grad():
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        start = time.time()
        encoder_model.set_width(alpha)
        for image, image_id in image_list:
            print(image_id)
            w = image.size[0]
            h = image.size[1]
            image = image.resize((640, 640))
            x = tfunc.to_tensor(image)
            x = x.unsqueeze(0)
            x = encoder_model(x)
        end = time.time()
        elapsed = (end-start)
        print("[alpha={}], Sample Processing Time: {}".format(alpha, elapsed))




