import json
import math
import os
import sys
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, '../common')
import constants
import jsonlines

class ResultManager:

    def __init__(self, results_filename):
        self.cocoGt = COCO(constants.annFile)
        self.results_filename = results_filename
        # If data file exists load them
        if not os.path.exists(results_filename):
            self.reset()

    def reset(self):
        if os.path.exists(self.results_filename):
            os.remove(self.results_filename)

        with open(self.results_filename, "w+") as f:
            pass

        os.chmod(self.results_filename, constants.safe_mode)

    def update(self, detections):
        with jsonlines.open(self.results_filename, mode='a') as writer:
            for d in detections:
                writer.write(d)

    def get(self):
        with open(self.results_filename, mode='r') as f:
            self.complete_results = json.load(f)

        os.chmod(self.results_filename, constants.safe_mode)
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

