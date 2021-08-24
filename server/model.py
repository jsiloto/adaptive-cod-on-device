import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
adaptive_cod_path = os.path.join(os.path.dirname(currentdir), 'adaptive-cod/src/')
sys.path.append(adaptive_cod_path)
from models import get_model
from myutils.common import file_util, yaml_util

# config_filename = os.path.join(currentdir, "adaptive-cod/config/acod/")
# config = yaml_util.load_yaml_file("")

def get_model():



    return None

