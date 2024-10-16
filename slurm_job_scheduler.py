import os
import time
from itertools import product

import yaml


def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)


def construct_args(params, combination):
    args = []
    for key, value in zip(params.keys(), combination):
        if key in ["scheduler", "intensity_aug"]:
            if not value:
                args.append(f"--no-{key}")
        else:
            args.append(f"--{key} {value}")
    return " ".join(args)


if __name__ == "__main__":
    config = load_config("config.yaml")
    params = config['params']

    for combination in product(*params.values()):
        command = f"sbatch aff_train.sh {construct_args(params, combination)}"
        print(f"Executing command: {command}")
        os.system(command)
        time.sleep(1)

    print("All jobs started")
