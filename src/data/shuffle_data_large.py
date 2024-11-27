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


def get_data_file_paths(flag, percent=1):
    data_files = glob.glob(f"/ssl-jet-vol-v3/JetClass/processed/raw/raw_{flag}_{percent}%/data/*")
    if percent == 100:
        data_files = glob.glob(f"/ssl-jet-vol-v3/JetClass/processed/raw/{flag}/data/*")
    return data_files
    
def load_data(data_files):
    data = []
    for i, file in enumerate(data_files):
        data += torch.load(file)
        print(f"--- loaded {file}")
    return data

def shuffle_and_save(data_file_paths, label_file_paths, save_dir_data, save_dir_label):
    assert len(data_file_paths) == len(label_file_paths), "Data and label files count must be equal."
    
    indices = list(range(len(data_file_paths)))
    random.shuffle(indices)  
    
    for j in range(0, len(indices), 10):
        print(j)
        batch_indices = indices[j:j+10]
        # print(batch_indices)
        data_batch_paths = [data_file_paths[index] for index in batch_indices]
        label_batch_paths = [label_file_paths[index] for index in batch_indices]
        
        data_content = load_data(data_batch_paths)
        label_content = load_data(label_batch_paths)
        data_torch = torch.stack(data_content)
        labels_torch = torch.stack(label_content)
        
        # Seed for reproducibility
        torch.manual_seed(42)
        
        # Generate shuffled indices
        jet_indices = torch.randperm(data_torch.size(0))
        # Apply shuffled indices to both data and labels
        data_torch_shuffled = data_torch[jet_indices]
        labels_torch_shuffled = labels_torch[jet_indices]

        # To ensure continuous indices for saved files
        save_indices = list(range(j, j+10))
        save_tensors_in_chunks(data_torch_shuffled, labels_torch_shuffled, save_dir_data, save_dir_label, save_indices)

        # Explicitly delete objects to free up memory
        del data_batch_paths, label_batch_paths, data_content, label_content, data_torch, labels_torch, data_torch_shuffled, labels_torch_shuffled
        gc.collect()  # Force garbage collection

def save_tensors_in_chunks(data_tensor, label_tensor, save_dir_data, save_dir_label, save_indices):
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
        torch.save(data_chunk, os.path.join(save_dir_data, f'data_{save_indices[i]}.pt'))
        torch.save(label_chunk, os.path.join(save_dir_label, f'label_{save_indices[i]}.pt'))
        print(f"saved file {save_indices[i]}")
        del data_chunk, label_chunk
        gc.collect()
    del data_tensor, label_tensor
    gc.collect()

def main(args):
    flag = args.flag
    percent = args.percent
    data_file_paths = get_data_file_paths(flag, percent)
    label_file_paths = [path.replace("data/data", "label/labels") for path in data_file_paths]
    if percent == 100:
        label_file_paths = [path.replace("data/", "label/labels_") for path in data_file_paths]

    save_dir = f"/ssl-jet-vol-v3/JetClass/processed/raw/raw_{flag}_{percent}%/shuffled"
    save_dir_data = f"{save_dir}/data"
    save_dir_label = f"{save_dir}/label"
    os.makedirs(save_dir_data, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    shuffle_and_save(data_file_paths, label_file_paths, save_dir_data, save_dir_label)


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
