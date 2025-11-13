import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import tqdm
import argparse
import csv

sys.path.append("../")

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# load custom modules required for jetCLR training
from src.modules.jet_augs import (
    rotate_jets,
    distort_jets,
    translate_jets,
    collinear_fill_jets,
)
from src.modules.transformer import Transformer
from src.modules.losses import contrastive_loss, align_loss, uniform_loss
from src.modules.perf_eval import get_perf_stats, linear_classifier_test
from src.modules.dataset import JetClassDataset

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
inpput_dim = 7
args.sbratio = 1
args.output_dim = 1000
args.model_dim = 1000
args.n_heads = 4
args.dim_feedforward = 1000
args.n_layers = 4
args.learning_rate = 0.00005
args.n_head_layers = 2
args.opt = "adam"
args.label = "JetClass1percent-test-new-image"
args.load_path = f"/ssl-jet-vol-v3/JetCLR/models/{args.label}/model_ep20.pt"
args.save_path = f"/ssl-jet-vol-v3/JetCLR/plots/cosine_similarity/{args.label}"
args.trs = True
args.rot = True
args.cf = True
args.ptd = True
args.mask = False
args.cmask = True
args.batch_size = 512
args.trsw = 0.1
args.ptst = 0.1
args.ptcm = 0.1
args.full_kinematics = True
args.percent = 1

os.makedirs(args.save_path, exist_ok=True)


def augmentation(args, x_i):
    x_i = x_i.cpu().numpy()
    # x_i has shape (batch_size, 7, n_constit)
    # dim 1 ordering: 'part_eta','part_phi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR'
    # extract the (pT, eta, phi) features for augmentations
    log_pT = x_i[:, 2, :]
    pT = np.where(log_pT != 0, np.exp(log_pT), 0)  # this handles zero-padding
    eta = x_i[:, 0, :]
    phi = x_i[:, 1, :]
    x_i = np.stack([pT, eta, phi], 1)  # (batch_size, 3, n_constit)
    time1 = time.time()
    x_i = rotate_jets(x_i)
    x_j = x_i.copy()
    if args.rot:
        x_j = rotate_jets(x_j)
    time2 = time.time()
    if args.cf:
        x_j = collinear_fill_jets(x_j)
        x_j = collinear_fill_jets(x_j)
    time3 = time.time()
    if args.ptd:
        x_j = distort_jets(x_j, strength=args.ptst, pT_clip_min=args.ptcm)
    time4 = time.time()
    if args.trs:
        x_j = translate_jets(x_j, width=args.trsw)
        x_i = translate_jets(x_i, width=args.trsw)
    time5 = time.time()
    # recalculate the rest of the features after augmentation
    pT_i = x_i[:, 0, :]
    eta_i = x_i[:, 1, :]
    phi_i = x_i[:, 2, :]
    pT_j = x_j[:, 0, :]
    eta_j = x_j[:, 1, :]
    phi_j = x_j[:, 2, :]
    # calculate the rest of the features
    # pT
    pT_log_i = np.where(pT_i != 0, np.log(pT_i), 0)
    pT_log_i = np.nan_to_num(pT_log_i, nan=0.0)
    pT_log_j = np.where(pT_j != 0, np.log(pT_j), 0)
    pT_log_j = np.nan_to_num(pT_log_j, nan=0.0)
    # pTrel
    pT_sum_i = np.sum(pT_i, axis=-1, keepdims=True)
    pT_sum_j = np.sum(pT_j, axis=-1, keepdims=True)
    pt_rel_i = pT_i / pT_sum_i
    pt_rel_j = pT_j / pT_sum_j
    pt_rel_log_i = np.where(pt_rel_i != 0, np.log(pt_rel_i), 0)
    pt_rel_log_i = np.nan_to_num(pt_rel_log_i, nan=0.0)
    pt_rel_log_j = np.where(pt_rel_j != 0, np.log(pt_rel_j), 0)
    pt_rel_log_j = np.nan_to_num(pt_rel_log_j, nan=0.0)
    # E
    E_i = pT_i * np.cosh(eta_i)
    E_j = pT_j * np.cosh(eta_j)
    E_log_i = np.where(E_i != 0, np.log(E_i), 0)
    E_log_i = np.nan_to_num(E_log_i, nan=0.0)
    E_log_j = np.where(E_j != 0, np.log(E_j), 0)
    E_log_j = np.nan_to_num(E_log_j, nan=0.0)
    # Erel
    E_sum_i = np.sum(E_i, axis=-1, keepdims=True)
    E_sum_j = np.sum(E_j, axis=-1, keepdims=True)
    E_rel_i = E_i / E_sum_i
    E_rel_j = E_j / E_sum_j
    E_rel_log_i = np.where(E_rel_i != 0, np.log(E_rel_i), 0)
    E_rel_log_i = np.nan_to_num(E_rel_log_i, nan=0.0)
    E_rel_log_j = np.where(E_rel_j != 0, np.log(E_rel_j), 0)
    E_rel_log_j = np.nan_to_num(E_rel_log_j, nan=0.0)
    # deltaR
    deltaR_i = np.sqrt(np.square(eta_i) + np.square(phi_i))
    deltaR_j = np.sqrt(np.square(eta_j) + np.square(phi_j))
    # stack them to obtain the final augmented data
    x_i = np.stack(
        [eta_i, phi_i, pT_log_i, E_log_i, pt_rel_log_i, E_rel_log_i, deltaR_i],
        1,
    )  # (batch_size, 6, n_constit)
    x_j = np.stack(
        [eta_j, phi_j, pT_log_j, E_log_j, pt_rel_log_j, E_rel_log_j, deltaR_j],
        1,
    )  # (batch_size, 6, n_constit)
    x_i = torch.Tensor(x_i).transpose(1, 2).to(args.device)
    x_j = torch.Tensor(x_j).transpose(1, 2).to(args.device)
    times = [time1, time2, time3, time4, time5]
    return x_i, x_j, times


args.n_heads = 4
args.opt = "adam"
args.learning_rate = 0.00005 * args.batch_size / 128
# define the global base device
world_size = torch.cuda.device_count()
if world_size:
    device = torch.device("cuda:0")
    for i in range(world_size):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
else:
    device = torch.device("cpu")
    print("Device: CPU", file=logfile, flush=True)
args.device = device
print("loading data")
# Initialize JetClass custom dataset
dataset_path = "/ssl-jet-vol-v3/JetClass/processed/raw"
test_dataset = JetClassDataset(
    dataset_path,
    flag="test",
    args=args,
    logfile=sys.stdout,
    load_labels=True,
)
input_dim = test_dataset.get_sample_shape()[0]
print(f"input_dim: {input_dim}")


args.batch_size = 160


def compute_avg_cosine_similarity_for_models(
    model_path_pattern, test_dataset, args, net, augmentation
):
    """
    Compute and return the average cosine similarity for each saved model.

    :param model_path_pattern: Pattern for the model paths, e.g., "/ssl-jet-vol-v3/JetCLR/models/{label}/model_ep{ep}.pt"
    :param test_dataset: The dataset to use for testing
    :param args: Arguments containing device, batch_size, etc.
    :param net: The neural network model (uninitialized)
    :param augmentation: The augmentation function to apply
    :return: A dictionary containing average cosine similarities for each epoch/model
    """
    avg_cosine_similarities_by_epoch = {}

    # Iterate over the models saved at every 10 epochs from 0 to 300
    for ep in range(0, 301, 10):
        print(ep)
        model_path = model_path_pattern.format(label=args.label, ep=ep)
        try:
            net.load_state_dict(torch.load(model_path, map_location=args.device))
            net.eval()

            # DataLoader setup
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
            )

            # Initialize tqdm
            total = int(len(test_dataset) / args.batch_size)
            pbar_t = tqdm.tqdm(
                test_loader,
                total=total // 10,
                desc=f"Inference for Epoch {ep}",
            )

            # Initialize dictionary for this epoch
            avg_cosine_sim = {}

            for i, (batch_data, batch_labels) in enumerate(pbar_t):
                # Process the batch here as described previously
                if i == total // 10:
                    break
                batch_data = batch_data.to(
                    args.device
                )  # Assuming shape (batch_size, 7, 128)
                batch_labels = batch_labels.to(
                    args.device
                )  # Assuming shape (batch_size, 10) for one-hot encoded labels
                net.optimizer.zero_grad()
                x_i, x_j, times = augmentation(args, batch_data)
                batch_size = x_i.shape[0]
                x_i = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask)
                x_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
                z_i = F.normalize(x_i, dim=1)
                z_j = F.normalize(x_j, dim=1)
                z = torch.cat([z_i, z_j], dim=0)
                similarity_matrix = F.cosine_similarity(
                    z.unsqueeze(1), z.unsqueeze(0), dim=2
                )
                sim_ij = torch.diag(similarity_matrix, batch_size)

                # Convert batch_labels from one-hot to indices for easier processing
                labels_indices = torch.argmax(batch_labels, dim=1)

                # Iterate through each label index and append the corresponding sim_ij value
                for label_idx in range(labels_indices.size(0)):
                    label = labels_indices[label_idx].item()
                    sim_value = sim_ij[label_idx].item()

                    if label not in avg_cosine_sim:
                        avg_cosine_sim[label] = []
                    avg_cosine_sim[label].append(sim_value)

            # After processing each batch, calculate the avg_cosine_sim for this model
            avg_cosine_sim_final = {}
            for label, sim_values in avg_cosine_sim.items():
                avg_cosine_sim_final[label] = np.mean(sim_values)

            # Store the result for this epoch
            avg_cosine_similarities_by_epoch[ep] = avg_cosine_sim_final

        except FileNotFoundError:
            print(f"Model file not found: {model_path}. Skipping...")
            continue

    return avg_cosine_similarities_by_epoch


# Example of how to call the function
net = Transformer(
    input_dim,
    args.model_dim,
    args.output_dim,
    args.n_heads,
    args.dim_feedforward,
    args.n_layers,
    args.learning_rate,
    args.n_head_layers,
    dropout=0.1,
    opt=args.opt,
    log=True,
)
net.to(device)

avg_similarities = compute_avg_cosine_similarity_for_models(
    model_path_pattern="/ssl-jet-vol-v3/JetCLR/models/{label}/model_ep{ep}.pt",
    test_dataset=test_dataset,
    args=args,
    net=net,
    augmentation=augmentation,
)
# Save as CSV
import json

with open(f"{args.save_path}/avg_sim.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for key, value in avg_similarities.items():
        # Serialize the dictionary value to a JSON string
        json_value = json.dumps(value)
        writer.writerow([key, json_value])
print(avg_similarities)

class_labels = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def plot_avg_cosine_similarities(args, avg_similarities):
    """
    Plots the average cosine similarities for each class across epochs.

    :param avg_similarities: Dictionary of average cosine similarities returned by
                             compute_avg_cosine_similarity_for_models function.
    """
    os.makedirs(args.save_path, exist_ok=True)
    # Assuming each entry in avg_similarities contains keys for each class label
    # and values are the average cosine similarity for that class at a specific epoch

    # Initialize a figure
    plt.figure(figsize=(10, 6))

    # For each class, plot its average similarity across epochs
    for i, class_label in enumerate(
        class_labels
    ):  # Assuming class labels are from 0 to 9
        epochs = sorted(avg_similarities.keys())
        avg_sims = [
            avg_similarities[ep][i] for ep in epochs if i in avg_similarities[ep]
        ]

        plt.plot(epochs, avg_sims, marker="o", label=class_label)

    plt.xlabel("Epoch")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Average Cosine Similarity per Class Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        f"{args.save_path}/avg_cosine_similarities.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


# Assuming `avg_similarities` is the dictionary returned by compute_avg_cosine_similarity_for_models function
plot_avg_cosine_similarities(args, avg_similarities)
