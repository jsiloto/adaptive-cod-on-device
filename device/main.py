import operator
from functools import reduce  # Required in Python 3
import time
from PIL import Image
import tqdm

from dataset import coco_image_list
from api import API


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def compute_image_list(api, image_list):
    api.delete_results()
    pbar = tqdm.tqdm(enumerate(image_list))
    for i, (image_path, image_id) in pbar:
        start = time.time()
        image = Image.open(image_path)
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
    image_list = coco_image_list(subval=False)[0:1]
    # image_list = [i for i in image_list if i[1] in ['000000127135.jpg', '000000127182.jpg']]
    # image_list = [i for i in image_list if i[1] in ['000000127092.jpg', '000000127182.jpg']]
    print(image_list)

    def run_at_alpha(width):
        print("############ Compute using Alpha: {:1.2f} ###################".format(width))
        api.set_width(width)
        result = compute_image_list(api, image_list)
        api.post_data("mock_device_{:03d}".format(int(width*100)), result)

    run_at_alpha(0.25)
    # run_at_alpha(0.50)
    # run_at_alpha(0.75)
    # run_at_alpha(1.00)
