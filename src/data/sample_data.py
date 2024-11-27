# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import argparse
import numpy as np
import awkward as ak
import uproot
import vector

vector.register_awkward()
import torch
import os
import os.path as osp
import glob
import os
import gc


def modify_path(path):
    """
    Given a path for a data file, e.g. path = '/ssl-jet-vol-v2/JetClass/processed/val/data/HToBB_123.pt',
    Constructs the path for the corresponding label file, e.g. new_path = '/ssl-jet-vol-v2/JetClass/processed/val/label/labels_HToBB_123.pt'
    """
    # Split the string into parts
    parts = path.split("/")

    # Replace 'data' with 'label' in the second-to-last part
    parts[-2] = "label"

    # Insert 'labels_' before the last part
    parts[-1] = "labels_" + parts[-1]

    # Join the parts back together
    new_path = "/".join(parts)
    return new_path


def main(args):
    """
    Samples a fraction of jets from all data files and saves them to a new directory.
    Shape: (100k, 7, 128)
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    label = args.label
    data_dir = f"/ssl-jet-vol-v2/JetClass/processed/raw/{label}"
    data_files = glob.glob(f"{data_dir}/data/*")
    # frac_lst = [1, 5, 10, 50]
    frac_lst = [1]
    total_samples = 100000  # 100k jets per file

    current_index = 0  # Keep track of where to insert data
    for frac in frac_lst:
        file_counter = 0
        print(f"Sampling {frac}% of data from `{label}` directory")
        processed_data_dir = (
            f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{label}_{frac}%_2/data"
        )
        processed_label_dir = (
            f"/ssl-jet-vol-v2/JetClass/processed/raw/raw_{label}_{frac}%_2/label"
        )
        os.system(
            f"mkdir -p {processed_data_dir} {processed_label_dir}"
        )  # -p: create parent dirs if needed, exist_ok

        # Assuming you know the total number of samples after sampling from all files
        data_shape = (total_samples, 7, 128)  # Example data shape
        label_shape = (total_samples, 10)  # Example label shape, adjust as needed

        # Pre-allocate tensors
        temp_sampled_data = torch.zeros(
            data_shape, dtype=torch.float32
        )  # Adjust dtype as needed
        temp_sampled_labels = torch.zeros(
            label_shape, dtype=torch.int64
        )  # Adjust dtype as needed

        for i, file in enumerate(data_files):
            data = torch.load(file)  # Assume shape is (N, 7, 128) for each file
            data_file_name = file.split("/")[-1].split(".")[0]
            print(f"--- loaded data file {i} {data_file_name} from `{label}` directory")

            label_path = modify_path(file)
            labels = torch.load(label_path)  # Assume shape is (N,) for each file

            # Calculate number of samples to take
            num_samples = int(frac / 100 * data.shape[0])
            # Generate random indices
            indices = torch.randperm(data.shape[0])[:num_samples]

            # Fill pre-allocated tensors
            # Calculate new end index based on current batch
            end_index = current_index + num_samples

            # Check if the temp tensors can accommodate this batch; if not, save and reset
            if end_index > temp_sampled_data.shape[0]:
                # Save the filled portion of the tensors
                torch.save(
                    temp_sampled_data[:current_index],
                    os.path.join(processed_data_dir, f"data_{file_counter}.pt"),
                )
                torch.save(
                    temp_sampled_labels[:current_index],
                    os.path.join(processed_label_dir, f"labels_{file_counter}.pt"),
                )
                file_counter += 1  # Increment file counter after saving
                print(f"----finished creating {file_counter} files")

                # Reset current index for the next batch of tensors
                current_index = 0
                end_index = num_samples  # As we are starting from 0 again

                # Reset current index for the next batch of tensors, only if not at the last file
                if i != len(data_files) - 1:
                    print("resetting  temp storage tensors")
                    temp_sampled_data.fill_(0)  # Resetting tensors for reuse
                    temp_sampled_labels.fill_(0)

            # Fill pre-allocated tensors with the current batch
            temp_sampled_data[current_index:end_index] = data[indices]
            temp_sampled_labels[current_index:end_index] = labels[indices]

            current_index = end_index  # Update current index

            # Condition to save at the last file
            if i == len(data_files) - 1:
                # Save the filled portion of the tensors
                torch.save(
                    temp_sampled_data[:current_index],
                    os.path.join(processed_data_dir, f"data_{file_counter}.pt"),
                )
                torch.save(
                    temp_sampled_labels[:current_index],
                    os.path.join(processed_label_dir, f"labels_{file_counter}.pt"),
                )
                file_counter += 1  # Increment file counter after saving
            # delete the data and labels to free up memory
            del data, labels
            gc.collect()

        # Reset for the next fraction, if necessary
        current_index = 0
        file_counter = 0  # Reset file counter for the next fraction
        print(f"Finished sampling and saving {frac}% of data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        type=str,
        action="store",
        default="train",
        help="train/val/test",
    )
    args = parser.parse_args()
    main(args)
