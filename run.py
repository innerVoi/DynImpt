import os, sys
import toml
import argparse
from munch import Munch, munchify
import time


PROJ_DIR = os.path.expanduser("./")
DATA_DIR = os.path.join(PROJ_DIR, "data")
LOGS_DIR = os.path.join(PROJ_DIR, "logs")
RESULTS_DIR = os.path.join(PROJ_DIR, "checkpoints")

sys.path.append(PROJ_DIR)
# from base_train import train
from mod_train import train
# from noise_train import train
from utils import update_config


def run(args:argparse, config:Munch) -> None:
    config.default.data_dir = DATA_DIR
    config.default.logs_dir = LOGS_DIR
    config.default.results_dir = RESULTS_DIR
    s = {'experiment': args.experiment, 'sigma': args.sigma, 'tau': args.tau, 'gamma': args.gamma, 'momentum': args.momentum}
    def _train_on_dataset_(dataset:str) -> None:
        best_test_acc = -1.0
        for seed in config.sweep.seeds:
            config.default.seed = seed
            test_acc = train(config, dataset, s)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

    if args.mnist:
        _train_on_dataset_("MNIST")
    
    if args.cifar10:
        _train_on_dataset_("CIFAR10")

    if args.cifar100:
        _train_on_dataset_("CIFAR100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="PreTraining")
    parser.add_argument("--cifar10", action="store_true", default=False,
    help="run cifar10 experiment")
    parser.add_argument("--cifar100", action="store_true", default=False, 
    help="run cifar100 experiment")
    parser.add_argument("--notes",   default=None)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    args, unknown = parser.parse_known_args()

    config = toml.load(os.path.join(PROJ_DIR,"config.toml"))
    config = update_config(unknown, config)
    config = munchify(config)

    run(args, config)


