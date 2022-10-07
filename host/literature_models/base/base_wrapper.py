from typing import List
from torch import nn


class BaseWrapper():

    @classmethod
    def get_mode_options(cls) -> List[int]:
        return cls.get_mode_options()

    def get_input_shape(self) -> (int, int, int):
        raise NotImplementedError()


    def get_printname(self) -> str:
        raise NotImplementedError()

    def get_reported_results(self, mode) -> (float, float):
        raise NotImplementedError("Should return (mAP, bandwidth-Bytes)")

    def generate_torchscript(self, out_dir) -> str:
        raise NotImplementedError("Should return the output .ptl file when implemented")

    def generate_metrics(self):
        raise NotImplementedError("Should return a dictionary with metrics data")
