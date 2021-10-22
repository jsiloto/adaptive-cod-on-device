import operator
from functools import reduce  # Required in Python 3
import time
from dataset import subval_image_list
from api import delete_results, get_results, offload, split_offload, post_results

base_url = 'http://0.0.0.0:5000/'


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def compute_image_list(image_list, compute_func):
    delete_results(base_url)
    for i, (image, image_id) in enumerate(image_list):
        start = time.time()
        # image, image_id = image_list[i]
        compute_func(base_url, image, image_id)
        end = time.time()
        elapsed = (end - start)
        print("[{}], Sample Processing Time: {}".format(image.size, elapsed))
    result = get_results(base_url)
    post_results(base_url, "test", result)
    print(result.content)


if __name__ == '__main__':
    image_list = subval_image_list()

    # print("###############################")
    # print("Computing using Full Offloading")
    # compute_image_list(image_list, compute_func=offload)

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
