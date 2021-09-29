def filter_detection(det):
    print(det)
    k = {
        "class": det['category_id'],
        "score": det['score'],
        "bbox": [int(i) for i in det["bbox"]]
    }
    print(k)
    return k


def detection2response(detections):
    response = [filter_detection(d) for d in detections]
    return {"data": response}
