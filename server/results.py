import json
import math
import os
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

safe_mode = 0o777  # LOL
annFile = '../resource/dataset/coco2017/annotations/instances_subval2017.json'


class ResultManager:

    def __init__(self, results_filename):
        self.cocoGt = COCO(annFile)
        self.complete_results = []
        self.results_filename = results_filename
        # If data file exists load them
        if os.path.exists(results_filename):
            with open(results_filename, "r") as f:
                self.complete_results = json.load(f)
        else:
            self.reset()

    def reset(self):
        self.complete_results = []
        if os.path.exists(self.results_filename):
            os.remove(self.results_filename)

        with open(self.results_filename, "w+") as f:
            json.dump(self.complete_results, f)

        os.chmod(self.results_filename, safe_mode)

    def update(self, detections):
        self.complete_results += detections

        with open(self.results_filename, "w") as f:
            json.dump(self.complete_results, f)

        os.chmod(self.results_filename, safe_mode)

    def get(self):
        cocoDt = self.cocoGt.loadRes(self.results_filename)
        cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
        imgIds = [i['image_id'] for i in self.complete_results]
        cocoEval.params.imgIds = imgIds

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats.tolist()
        stats = [math.ceil(v * 1000) / 1000 for v in stats]
        return stats

