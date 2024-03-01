#!/bin/env python3.7

# load standard python modules
import sys
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, "../src")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import argparse
import copy
import tqdm
import gc

# load torch modules
import torch

# define the global base device
world_size = torch.cuda.device_count()
if world_size:
    device = torch.device("cuda:0")
    for i in range(world_size):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
else:
    device = torch.device("cpu")
    print("Device: CPU", flush=True)


def load_data(flag, percent=1):
    data_files = glob.glob(
        f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{flag}_{percent}%/data/*"
    )
    # print(data_files)

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(
            f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{flag}_{percent}%/data/data_{i}.pt"
        )
        print(f"--- loaded file {i} from `{flag}` directory")
    return data


def load_labels(flag, percent=1):
    data_files = glob.glob(
        f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{flag}_{percent}%/label/*"
    )

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(
            f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{flag}_{percent}%/label/labels_{i}.pt"
        )
        print(f"--- loaded label file {i} from `{flag}` directory")

    return data


def save_tensors_in_chunks(data_tensor, label_tensor, save_dir_data, save_dir_label):
    # Ensure the save directories exist
    os.makedirs(save_dir_data, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    # Calculate the number of samples per file
    total_samples = data_tensor.size(0)
    samples_per_file = 100000
    num_files = total_samples // samples_per_file

    if total_samples < samples_per_file:
        samples_per_file = total_samples
        num_files = 1

    for i in range(num_files):
        start_index = i * samples_per_file
        # Handle the last file which might have more samples due to rounding
        end_index = (i + 1) * samples_per_file if i < num_files - 1 else total_samples

        # Extract the current chunk for data and labels
        data_chunk = torch.empty((samples_per_file, 7, 128))
        label_chunk = torch.empty((samples_per_file, 10))
        data_chunk[:] = data_tensor[start_index:end_index]
        label_chunk[:] = label_tensor[start_index:end_index]

        # Save the current chunk to file
        torch.save(data_chunk, os.path.join(save_dir_data, f"data_{i}.pt"))
        torch.save(label_chunk, os.path.join(save_dir_label, f"label_{i}.pt"))
        print(f"saved file {i}")


def main(args):
    flag = args.flag
    percent = args.percent
    data = load_data(flag, percent)
    labels = load_labels(flag, percent)
    data_torch = torch.stack(data)
    labels_torch = torch.stack(labels)

    # Seed for reproducibility
    torch.manual_seed(42)

    # Generate shuffled indices
    indices = torch.randperm(data_torch.size(0))

    # Apply shuffled indices to both data and labels
    data_torch_shuffled = data_torch[indices]
    labels_torch_shuffled = labels_torch[indices]

    save_dir = f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{flag}_{percent}%/shuffled"
    save_dir_data = f"{save_dir}/data"
    save_dir_label = f"{save_dir}/label"
    os.makedirs(save_dir_data, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    # Call the function with your tensors and desired save directories
    save_tensors_in_chunks(
        data_torch_shuffled, labels_torch_shuffled, save_dir_data, save_dir_label
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flag",
        type=str,
        action="store",
        default="train",
        help="train/val/test",
    )
    parser.add_argument(
        "--percent",
        type=int,
        action="store",
        default=5,
        help="percent of data to load",
    )
    args = parser.parse_args()
    main(args)
