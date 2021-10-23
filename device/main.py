import operator
from functools import reduce  # Required in Python 3
import time

import tqdm

from dataset import subval_image_list
from api import API


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def compute_image_list(api, image_list):
    api.delete_results()
    pbar = tqdm.tqdm(enumerate(image_list))
    for i, (image, image_id) in pbar:
        start = time.time()
        api.split_offload(image, image_id)
        end = time.time()
        elapsed = (end - start)
        message = "[{}], Sample Processing Time: {:2.3f}s".format(image.size, elapsed)
        pbar.set_description(message)

    result = api.get_results()
    return result



if __name__ == '__main__':
    base_url = 'http://0.0.0.0:5000/'
    api = API(base_url)
    image_list = subval_image_list()

    def run_at_alpha(width):
        print("############ Compute using Alpha: {:1.2f} ###################".format(width))
        api.set_width(width)
        result = compute_image_list(api, image_list)
        api.post_data("mock_device_{:03d}".format(int(width*100)), result)

    run_at_alpha(0.25)
    run_at_alpha(0.50)
    run_at_alpha(0.75)
    run_at_alpha(1.00)
