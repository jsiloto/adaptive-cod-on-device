import jsonlines
import numpy as np

def read_jsonlines_file(filename):
    ignore_start = 5

    def get_values(key):
        with jsonlines.open(filename) as reader:
            values = [obj[key] for obj in reader]
        values = values[ignore_start:]
        return values

    results = {}
    results["bw"] = get_values("bw")
    results["e2e"] = get_values("e2e")
    results["mAP"] = get_values("map")
    results["deadline"] = get_values("deadline")
    time = get_values("time")
    results["time"] = [t - time[0] for t in time]
    results["mode"] = get_values("mode")
    results["compute"] = [m // 10 for m in results["mode"]]
    results["bits"] = [m % 10 for m in results["mode"]]

    return results


def average_results(results):
    new_results = {}
    for key, values in results.items():
        new_results[key] = np.average(values)

    return new_results