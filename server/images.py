import os
import sys
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from models import affine
sys.path.insert(0, '../common')
import constants
sys.path.insert(0, './yolov5')
from yolov5.models.common import Detections


safe_mode = 0o777  # LOL
annFile = '../resource/dataset/coco2017/annotations/instances_subval2017.json'
dataset_path = "/workspace/resource/dataset/coco2017/val2017/"

class ImageManager:
    def __init__(self, ground_truth_image, prediction_image):
        self.cocoGt = COCO(annFile)
        self.ground_truth_image = ground_truth_image
        self.prediction_image = prediction_image

    def update_ground_truth(self, image_id):
        image_filename = "{:012d}.jpg".format(image_id)
        path = os.path.join(dataset_path, image_filename)
        image_id = int(image_filename.split('.jpg')[0])
        image = Image.open(path)
        orig_w, orig_h = image.size

        ids = self.cocoGt.getAnnIds(imgIds=[image_id])
        id_vals = self.cocoGt.loadAnns(ids=ids)

        vals = [{'bbox': affine(v['bbox'], orig_w, orig_h, size=768),
                 'category_id': v['category_id']} for v in id_vals]

        image = [np.asarray(image)]
        y = [torch.tensor([v['bbox'][:] + [1.0, v['category_id'] - 1] for v in vals])]

        a = Detections(image, y, path, [], constants.class_names, image[0].shape)
        a.render()
        Image.fromarray(a.imgs[0]).save(self.ground_truth_image)

    def update_prediction(self, image_id, results):
        image_filename = "{:012d}.jpg".format(image_id)
        path = os.path.join(dataset_path, image_filename)
        im = Image.open(path)
        results.imgs = [np.asarray(im)]
        results.render()
        im = Image.fromarray(results.imgs[0])
        im.save(self.prediction_image)
