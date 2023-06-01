import jsonlines
import numpy as np
import pandas as pd


def read_sota_old():
    df = pd.read_csv("rpi4_cpus4.csv")

    df['KB'] = df['KB'].div(1000).round(1)
    df['model'] = df["model"].replace("assine2022b", "ours", regex=True)
    df['ms'] = df['ms'].round(0).astype(int)
    df['mAP'] = df['mAP'].round(1)
    df = df.set_index("model")

    df = df.drop(index="ours_12")
    df = df.drop(index="ours_13")
    df = df.drop(index="ours_21")
    df = df.drop(index="ours_23")
    df = df.drop(index="ours_31")
    df = df.drop(index="ours_32")
    df = df.drop(index="ours_41")
    df = df.drop(index="ours_42")
    df = df.drop(index="ours_43")

    return df


def read_jsonlines_file_old(filename):
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
    results["compute"] = [int(m // 10) for m in results["mode"]]
    results["bits"] = [int(m % 10) for m in results["mode"]]
    # results['communication'] = get_values('rec_time')

    return results


def average_results_old(results):
    new_results = {}
    for key, values in results.items():
        new_results[key] = round(np.average(values), 1)

    new_results["mode"] = int(new_results["mode"])

    return new_results
