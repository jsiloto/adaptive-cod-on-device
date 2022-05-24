from typing import List
from torch import nn

class BaseWrapper():

    @classmethod
    def get_mode_options(cls) -> List[str]:
        return cls.get_mode_options()

    def get_printname(self) -> str:
        raise NotImplementedError()

    def generate_torchscript(self, out_dir) -> str:
        raise NotImplementedError("Should return the output .ptl file when implemented")

    def generate_metrics(self, out_dir) -> str:
        raise NotImplementedError("Should return the output .csv file when implemented")

