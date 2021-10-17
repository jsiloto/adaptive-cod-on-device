import os
import random
from PIL import Image




def random_image_list(num_images):
    image_list = []
    for i in range(num_images):
        image_id = random.choice(os.listdir("../resource/dataset/coco2017/val2017")) #change dir name to whatever
        image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
        image = Image.open(image_path)
        image_list.append((image, image_id))
    return image_list

def subval_image_list():
    image_list = []
    image_ids = os.listdir("../resource/dataset/coco2017/subval2017")
    for image_id in image_ids:
        image_path = os.path.join("../resource/dataset/coco2017/subval2017/", image_id)
        image = Image.open(image_path)
        image_list.append((image, image_id))
    return image_list

def single_image():
    image_id = "000000133244.jpg"
    image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
    image = Image.open(image_path)
    return image