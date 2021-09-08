import numpy as np
import requests
import torchvision.transforms.functional as tfunc
from urllib.parse import urljoin


def delete_results(base_url):
    url = urljoin(base_url, "results")
    res = requests.delete(url=url)
    return res

def get_results(base_url):
    url = urljoin(base_url, "results")
    res = requests.get(url=url)
    return res


def compute_image(base_url, image, image_id):
    url = urljoin(base_url, "compute")
    w = image.size[0]
    h = image.size[1]

    print(image.size)
    image = image.resize((640, 640))
    x = tfunc.to_tensor(image)
    x = (x * 255).numpy().astype(np.uint8)
    res = requests.post(url='http://0.0.0.0:5000/compute',
                        data=x.tobytes(),
                        headers={'Content-Type': 'application/octet-stream',
                                 "image_id": image_id,
                                 "w": str(w),
                                 "h": str(h)})

    return res