import os
import torch
import random
import numpy as np
from pathlib import Path
from munch import Munch
from typing import Dict


def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def create_dir_for_file(file_path:str) -> None:
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def update_config(unknown, config) -> Dict:
    for arg in unknown:
        if arg.startswith(("-", "--")):
            k, v = arg.split('=')
            k = k.replace("--", "")
            k = k.replace("-", "_")
            assert k in config['default'], f"unknown arg: {k=}"
            v_new = type(config['default'][k])(eval(v))
            print(f"Overwriting hps.{k} from {config['default'][k]} to {v_new}")
            config['default'][k] = v_new

    return config


def update_munch_config(config1:Munch, config2:Munch) -> Munch:
    for key in config2.keys():
        config1[key] = config2[key]
    return config1
