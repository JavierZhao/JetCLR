#!/bin/env python3.7

# load standard python modules
import sys

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
from contextlib import contextmanager

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

# load custom modules required for jetCLR training
# from src.modules.jet_augs import (
#     rotate_jets,
#     distort_jets,
#     translate_jets,
#     collinear_fill_jets,
# )
from src.modules.transformer import Transformer
from src.modules.losses import (
    contrastive_loss,
    align_loss,
    uniform_loss,
    contrastive_loss_new,
)
from src.modules.perf_eval import get_perf_stats, linear_classifier_test
from src.modules.dataset import JetClassDataset

# set the number of threads that pytorch will use
torch.set_num_threads(2)


def augmentation_cpu(args, x_i):
    x_i = x_i.cpu().numpy()
    # x_i has shape (batch_size, 7, n_constit)
    # dim 1 ordering: 'part_eta','part_phi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR'
    # extract the (pT, eta, phi) features for augmentations
    log_pT = x_i[:, 2, :]
    pT = np.where(log_pT != 0, np.exp(log_pT), 0)  # this handles zero-padding
    eta = x_i[:, 0, :]
    phi = x_i[:, 1, :]
    x_i = np.stack([pT, eta, phi], 1)  # (batch_size, 3, n_constit)
    x_i = rotate_jets(x_i)
    x_j = x_i.copy()
    if args.rot:
        x_j = rotate_jets(x_j)
    if args.cf:
        x_j = collinear_fill_jets(x_j)
        x_j = collinear_fill_jets(x_j)
    if args.ptd:
        x_j = distort_jets(x_j, strength=args.ptst, pT_clip_min=args.ptcm)
    if args.trs:
        x_j = translate_jets(x_j, width=args.trsw)
        x_i = translate_jets(x_i, width=args.trsw)
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
    # deltaR_i = np.sqrt(np.square(eta_i) + np.square(phi_i))
    # deltaR_j = np.sqrt(np.square(eta_j) + np.square(phi_j))
    # stack them to obtain the final augmented data
    x_i = np.stack(
        [
            eta_i,
            phi_i,
            pT_log_i,
            E_log_i,
            pt_rel_log_i,
            E_rel_log_i,
        ],
        1,
    )  # (batch_size, 6, n_constit)
    x_j = np.stack(
        [
            eta_j,
            phi_j,
            pT_log_j,
            E_log_j,
            pt_rel_log_j,
            E_rel_log_j,
        ],
        1,
    )  # (batch_size, 6, n_constit)
    x_i = torch.Tensor(x_i).transpose(1, 2).to(args.device)
    x_j = torch.Tensor(x_j).transpose(1, 2).to(args.device)
    return x_i, x_j


def augmentation_gpu(args, x_i):
    device = args.device
    # Assuming x_i is a PyTorch tensor on the appropriate device
    # x_i has shape (batch_size, 7, n_constit)
    # Extract the (pT, eta, phi) features for augmentations
    log_pT = x_i[:, 2, :]
    pT = torch.where(log_pT != 0, torch.exp(log_pT), torch.zeros_like(log_pT))
    eta = x_i[:, 0, :]
    phi = x_i[:, 1, :]
    x_i = torch.stack([pT, eta, phi], dim=1)  # (batch_size, 3, n_constit)

    x_i = rotate_jets(x_i, device)
    x_j = x_i.clone()
    if args.rot:
        x_j = rotate_jets(x_j, device)
    if args.cf:
        x_j = collinear_fill_jets(x_j, device)
        x_j = collinear_fill_jets(x_j, device)
    if args.ptd:
        x_j = distort_jets(x_j, device, strength=args.ptst, pT_clip_min=args.ptcm)
    if args.trs:
        x_j = translate_jets(x_j, device, width=args.trsw)
        x_i = translate_jets(x_i, device, width=args.trsw)

    torch.cuda.synchronize()  # Ensure all operations are completed

    # Recalculate the rest of the features after augmentation
    # pT logs
    # pT_log_i = torch.log(torch.clamp(x_i[:, 0, :], min=1e-6))
    # pT_log_j = torch.log(torch.clamp(x_j[:, 0, :], min=1e-6))
    pT_log_i = torch.where(x_i[:, 0, :] != 0, torch.log(x_i[:, 0, :]), 0)
    pT_log_j = torch.where(x_j[:, 0, :] != 0, torch.log(x_j[:, 0, :]), 0)

    # pTrel
    pT_sum_i = x_i[:, 0, :].sum(dim=-1, keepdim=True)
    pT_sum_j = x_j[:, 0, :].sum(dim=-1, keepdim=True)
    pt_rel_i = x_i[:, 0, :] / pT_sum_i
    pt_rel_j = x_j[:, 0, :] / pT_sum_j
    pt_rel_log_i = torch.log(torch.clamp(pt_rel_i, min=1e-6))
    pt_rel_log_j = torch.log(torch.clamp(pt_rel_j, min=1e-6))

    # E
    E_i = x_i[:, 0, :] * torch.cosh(x_i[:, 1, :])
    E_j = x_j[:, 0, :] * torch.cosh(x_j[:, 1, :])
    E_log_i = torch.log(torch.clamp(E_i, min=1e-6))
    E_log_j = torch.log(torch.clamp(E_j, min=1e-6))

    # Erel
    E_sum_i = E_i.sum(dim=-1, keepdim=True)
    E_sum_j = E_j.sum(dim=-1, keepdim=True)
    E_rel_i = E_i / E_sum_i
    E_rel_j = E_j / E_sum_j
    E_rel_log_i = torch.log(torch.clamp(E_rel_i, min=1e-6))
    E_rel_log_j = torch.log(torch.clamp(E_rel_j, min=1e-6))

    # Stack them to obtain the final augmented data
    x_i = torch.stack(
        [
            x_i[:, 1, :],  # eta_i
            x_i[:, 2, :],  # phi_i
            pT_log_i,
            E_log_i,
            pt_rel_log_i,
            E_rel_log_i,
        ],
        1,
    ).transpose(
        1, 2
    )  # (batch_size, 6, n_constit)

    x_j = torch.stack(
        [
            x_j[:, 1, :],  # eta_j
            x_j[:, 2, :],  # phi_j
            pT_log_j,
            E_log_j,
            pt_rel_log_j,
            E_rel_log_j,
        ],
        1,
    ).transpose(
        1, 2
    )  # (batch_size, 6, n_constit)
    return x_i, x_j


class_labels = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def plot_avg_cosine_similarities(args, avg_similarities):
    """
    Plots the average cosine similarities for each class across epochs, using a NumPy array format for avg_similarities,
    but only for epochs where the similarity is non-zero.

    :param args: Arguments containing save_plot_path.
    :param avg_similarities: NumPy array of average cosine similarities with shape [number_of_epochs, number_of_classes].
    """
    os.makedirs(args.save_plot_path, exist_ok=True)
    # Initialize a figure
    plt.figure(figsize=(10, 6))

    epochs = np.arange(0, avg_similarities.shape[0])

    # For each class, plot its average similarity across epochs if it's non-zero
    for i, class_label in enumerate(class_labels):
        avg_sims = avg_similarities[:, i]
        # Filter epochs and similarities for non-zero values
        non_zero_epochs = epochs[avg_sims != 0]
        non_zero_sims = avg_sims[avg_sims != 0]

        if (
            non_zero_sims.size > 0
        ):  # Check if there are any non-zero similarities to plot
            plt.plot(non_zero_epochs, non_zero_sims, marker="o", label=class_label)

    plt.xlabel("Epoch")
    plt.ylabel("Average Cosine Similarity")
    plt.title("Average Cosine Similarity per Class Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(args.save_plot_path, "avg_cosine_similarities.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


activations = {}
first_nan_inf_detected = False  # Flag to track if NaN or inf has been recorded


def save_activation(args, name):
    def hook(model, input, output):
        if type(output) is tuple:
            output = output[0]
        global first_nan_inf_detected
        if not first_nan_inf_detected:  # Check if NaN or inf has already been detected
            activations[name] = output.detach()
            if torch.isnan(output).any() or torch.isinf(output).any():
                first_nan_inf_detected = (
                    True  # Set the flag to prevent further recordings
                )
                print(f"Hook NaN or inf detected in {name}")
                print(
                    f"Hook NaN or inf detected in {name}", file=args.logfile, flush=True
                )

    return hook


@contextmanager
def optional_profiling(active, *args, **kwargs):
    if active:
        with profile(*args, **kwargs) as prof:
            yield prof
    else:
        yield None


def main(args):
    t0 = time.time()

    args.logfile = f"/ssl-jet-vol-v3/JetCLR/logs/JetClass/{args.person}-simCLR-{args.label}-log.txt"
    args.save_plot_path = f"/ssl-jet-vol-v3/JetCLR/plots/cosine_similarity/{args.label}"
    args.n_heads = 4
    args.opt = "adam"
    # args.learning_rate = 0.00005 * args.batch_size / 128
    args.learning_rate = args.base_lr * args.batch_size / 128

    # initialise logfile
    logfile = open(args.logfile, "a")
    if args.continue_training:
        print("-----------------------------------", file=logfile, flush=True)
        print("-----------------------------------", file=logfile, flush=True)
        print("continuing training", file=logfile, flush=True)
        print("-----------------------------------", file=logfile, flush=True)
        print("-----------------------------------", file=logfile, flush=True)
    print("logfile initialised", file=logfile, flush=True)
    if args.aug_device == "cpu":
        augmentation = augmentation_cpu
        print("Running augmentations on cpu ", file=logfile, flush=True)
    else:
        augmentation = augmentation_gpu
        print("Running augmentations on cuda ", file=logfile, flush=True)
    # print number of workers
    print("number of workers: " + str(args.n_workers), file=logfile, flush=True)

    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(
                f"Device {i}: {torch.cuda.get_device_name(i)}", file=logfile, flush=True
            )
    else:
        device = torch.device("cpu")
        print("Device: CPU", file=logfile, flush=True)
    args.device = device

    # set up results directory
    base_dir = "/ssl-jet-vol-v3/JetCLR/models/"
    expt_tag = args.label
    expt_dir = base_dir + "JetClass" + expt_tag + "/"

    # check if experiment already exists and is not empty
    if not args.continue_training:
        if os.path.isdir(expt_dir) and os.listdir(expt_dir):
            sys.exit(
                "ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake"
            )
        else:
            # This will create the directory if it does not exist or if it is empty
            os.makedirs(expt_dir, exist_ok=True)
    else:
        if not os.path.isdir(expt_dir) or not os.listdir(expt_dir):
            sys.exit(
                "ERROR: experiment does not exist or is empty, cannot continue training"
            )
    print("experiment: " + str(args.label), file=logfile, flush=True)

    print("loading data")
    # Initialize JetClass custom dataset
    dataset_path = "/ssl-jet-vol-v3/JetClass/processed/raw"
    train_dataset = JetClassDataset(
        dataset_path,
        flag="train",
        args=args,
        logfile=logfile,
        load_labels=False,
    )
    val_dataset = JetClassDataset(
        dataset_path,
        flag="val",
        args=args,
        logfile=logfile,
        load_labels=False,
    )
    test_dataset = JetClassDataset(
        dataset_path,
        flag="test",
        args=args,
        logfile=logfile,
        load_labels=True,
    )
    # load the test data
    # test_dat_in, test_lab_in = test_dataset.fetch_entire_dataset()
    # test_dat_1 = test_dat_in[: int(len(test_dat_in) / 2)]
    # test_lab_1 = test_lab_in[: int(len(test_lab_in) / 2)]
    # test_dat_2 = test_dat_in[int(len(test_dat_in) / 2) :]
    # test_lab_2 = test_lab_in[int(len(test_lab_in) / 2) :]
    # del test_dat_in, test_lab_in
    # gc.collect()

    input_dim = train_dataset.get_sample_shape()[0]
    print(f"input_dim: {input_dim}")

    # print data dimensions
    print(
        "Training data shape: " + str(train_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    print(
        "Validation data shape: " + str(val_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    print(
        "Testing data shape: " + str(test_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    print(
        "Testing labels shape: " + str(test_dataset.get_labels_shape()),
        flush=True,
        file=logfile,
    )

    t1 = time.time()

    print(
        "time taken to load data: " + str(np.round(t1 - t0, 2)) + " seconds",
        flush=True,
        file=logfile,
    )

    # set-up parameters for the LCT
    linear_input_size = args.output_dim
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 128  # we can try a larger batch size for the LCT

    print(
        "--- contrastive learning transformer network architecture ---",
        flush=True,
        file=logfile,
    )
    print("model dimension: " + str(args.model_dim), flush=True, file=logfile)
    print("number of heads: " + str(args.n_heads), flush=True, file=logfile)
    print(
        "dimension of feedforward network: " + str(args.dim_feedforward),
        flush=True,
        file=logfile,
    )
    print("number of layers: " + str(args.n_layers), flush=True, file=logfile)
    print("number of head layers: " + str(args.n_head_layers), flush=True, file=logfile)
    print("optimiser: " + str(args.opt), flush=True, file=logfile)
    print(f"use mask: {True if args.mask else False}", flush=True, file=logfile)
    print(
        f"use continuous mask: {True if args.cmask else False}",
        flush=True,
        file=logfile,
    )
    print("\n--- hyper-parameters ---", flush=True, file=logfile)
    print("learning rate: " + str(args.learning_rate), flush=True, file=logfile)
    print("batch size: " + str(args.batch_size), flush=True, file=logfile)
    print("temperature: " + str(args.temperature), flush=True, file=logfile)
    print("\n--- symmetries/augmentations ---", flush=True, file=logfile)
    print("rotations: " + str(args.rot), flush=True, file=logfile)
    print("low pT smearing: " + str(args.ptd), flush=True, file=logfile)
    print("pT smearing clip parameter: " + str(args.ptcm), flush=True, file=logfile)
    print("translations: " + str(args.trs), flush=True, file=logfile)
    print("translations width: " + str(args.trsw), flush=True, file=logfile)
    print(
        "Number of input features per particle: " + str(input_dim),
        flush=True,
        file=logfile,
    )
    print("---------------", flush=True, file=logfile)

    # initialise the network
    print("\ninitialising the network", flush=True, file=logfile)
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
        eps=args.eps,
    )

    if args.continue_training:
        print("Loading model from checkpoint", flush=True, file=logfile)
        model_load_path = expt_dir + "model_last.pt"
        net.load_state_dict(torch.load(model_load_path))
        # check if optimizer state is saved
        optimizer_load_path = expt_dir + "optimizer_last.pt"
        if os.path.isfile(optimizer_load_path):
            net.optimizer.load_state_dict(torch.load(optimizer_load_path))

    # send network to device
    net.to(device)
    print(net, flush=True, file=logfile)

    if args.use_hook:
        print("Registering forward hooks", flush=True, file=logfile)
        for name, module in net.named_modules():
            # if isinstance(module, (nn.Linear, nn.LayerNorm, nn.TransformerEncoder)):
            module.register_forward_hook(save_activation(args, name))

    # set learning rate scheduling, if required
    # SGD with cosine annealing
    if args.opt == "sgdca":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            net.optimizer, 15, T_mult=2, eta_min=0, last_epoch=-1, verbose=False
        )
    # SGD with step-reduced learning rates
    if args.opt == "sgdslr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            net.optimizer, 100, gamma=0.6, last_epoch=-1, verbose=False
        )

    # THE TRAINING LOOP
    print(
        "starting training loop, running for " + str(args.n_epochs) + " epochs",
        flush=True,
        file=logfile,
    )
    print("---------------", flush=True, file=logfile)

    # initialise lists for storing training stats
    auc_epochs = []
    imtafe_epochs = []
    losses = []
    losses_val = []
    loss_align_epochs = []
    loss_uniform_epochs = []

    # cosine annealing requires per-batch calls to the scheduler, we need to know the number of batches per epoch
    if args.opt == "sgdca":
        # number of iterations per epoch
        iters = int(len(train_dataset) / args.batch_size)
        print("number of iterations per epoch: " + str(iters), flush=True, file=logfile)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    num_classes = 10  # Assuming classes are labeled 0 through 9
    average_similarities = np.zeros((args.n_epochs, num_classes))

    l_val_best = 1000000  # initialise the best validation loss
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"./logs/profile/{args.label}")

    # Automatic Mixed Precision
    # turn args.use_amp into bool
    args.use_amp = args.use_amp == 1
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    if args.use_amp and args.continue_training:
        # check if the scaler state is saved
        if os.path.isfile(expt_dir + "scaler_last.pt"):
            scaler.load_state_dict(torch.load(expt_dir + "scaler_last.pt"))

    profile_active = True
    for epoch in range(args.n_epochs):
        # initialise timing stats
        te0 = time.time()

        # initialise lists to store batch stats
        loss_align_e = []
        loss_uniform_e = []
        losses_e = []
        losses_e_val = []

        # initialise timing stats
        td_aug = 0
        td_forward = 0
        td_loss = 0
        td_backward = 0

        t_cos_sim_start = time.time()
        # calculate the average cosine similarity for each class
        # initialize the arrays to store the average cosine similarities
        if not args.profile:
            print("Calculating average cosine similarities", flush=True, file=logfile)
            avg_similarities = np.zeros(num_classes)
            counts = np.zeros(num_classes, dtype=int)
            with torch.no_grad():
                net.eval()
                total = int(len(test_dataset) / args.batch_size)
                pbar_t = tqdm.tqdm(
                    test_loader,
                    total=total // 10,
                    desc=f"Inference for Epoch {epoch}",
                )

                for i, (batch_data, batch_labels) in enumerate(pbar_t):
                    if i == total // 10:
                        break
                    batch_data = batch_data.to(
                        args.device
                    )  # Assuming shape (batch_size, 7, 128)
                    batch_labels = batch_labels.to(
                        args.device
                    )  # Assuming shape (batch_size, 10) for one-hot encoded labels
                    x_i, x_j = augmentation(args, batch_data)
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

                        avg_similarities[label] += sim_value
                        counts[label] += 1

                average_similarities[epoch, :] = np.divide(
                    avg_similarities, counts, where=counts != 0
                )
            # save average similarities
            if epoch != 0:
                np.save(expt_dir + "average_similarities.npy", average_similarities)

            # plot the average cosine similarities
            plot_avg_cosine_similarities(args, average_similarities)
            t_cos_sim_end = time.time()
            td_cos_sim = t_cos_sim_end - t_cos_sim_start

        # the inner loop goes through the dataset batch by batch
        # augmentations of the jets are done on the fly
        with optional_profiling(
            profile_active,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=3, warmup=1, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(writer.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            net.train()
            total_train = int(len(train_dataset) / args.batch_size)
            pbar_t = tqdm.tqdm(
                train_loader,
                total=total_train,
                desc="Training",
            )

            for batch_num, batch in enumerate(pbar_t):
                time1 = time.time()
                batch = batch.to(args.device)  # shape (batch_size, 7, 128)
                x_i, x_j = augmentation(args, batch)
                time2 = time.time()
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=args.use_amp
                ):
                    # with torch.autocast(device_type="cuda", enabled=False):
                    z_i = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask)
                    z_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
                    time3 = time.time()
                    # calculate the alignment and uniformity loss for each batch
                    loss_align = align_loss(z_i, z_j)
                    loss_uniform_zi = uniform_loss(z_i)
                    loss_uniform_zj = uniform_loss(z_j)
                    with torch.no_grad():
                        loss_align_e.append(loss_align.detach())
                        loss_uniform_e.append(
                            (loss_uniform_zi.detach() + loss_uniform_zj.detach()) / 2
                        )
                    time4 = time.time()

                    # compute the loss, back-propagate, and update scheduler if required
                    if args.new_loss:
                        loss = contrastive_loss_new(z_i, z_j, args.temperature).to(
                            device
                        )
                    else:
                        loss = contrastive_loss(z_i, z_j, args.temperature).to(device)

                scaler.scale(loss).backward()
                # do gradient clipping
                if args.max_grad_norm > 0:
                    scaler.unscale_(net.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        net.parameters(), args.max_grad_norm
                    )
                    writer.add_scalar(
                        "Loss/grad_norm",
                        grad_norm,
                        epoch * len(train_loader) + batch_num,
                    )
                scaler.step(net.optimizer)
                scaler.update()
                writer.add_scalar(
                    "Loss/scaler_scale",
                    scaler._scale,
                    epoch * len(train_loader) + batch_num,
                )
                net.optimizer.zero_grad(set_to_none=args.set_to_none)
                if args.opt == "sgdca":
                    scheduler.step(epoch + i / iters)
                losses_e.append(loss.detach())
                time5 = time.time()
                pbar_t.set_description(f"Training loss: {loss:.4f}")
                #  check for NaNs and Infs in the gradients
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)):
                            print(f"NaN gradients in {name}")
                        if torch.any(torch.isinf(param.grad)):
                            print(f"Inf gradients in {name}")

                # update timing stats
                td_aug += time2 - time1
                td_forward += time3 - time2
                td_loss += time4 - time3
                td_backward += time5 - time4

                # Log training loss to TensorBoard
                writer.add_scalar(
                    "Loss/train", loss.item(), epoch * len(train_loader) + batch_num
                )
                if profile_active:
                    prof.step()
                if batch_num == 100 and args.profile:
                    break

            if profile_active:
                profile_active = False  # Only profile the first epoch

        losses_e_tensor = torch.stack(losses_e).cpu().numpy()
        loss_e = np.mean(losses_e_tensor)
        losses.append(loss_e)
        # summarise alignment and uniformity stats
        losses_align_array = torch.stack(loss_align_e).cpu().numpy()
        losses_uniform_array = torch.stack(loss_uniform_e).cpu().numpy()
        loss_align_epochs.append(np.mean(losses_align_array))
        loss_uniform_epochs.append(np.mean(losses_uniform_array))
        print(
            "alignment: "
            + str(loss_align_epochs[-1])
            + ", uniformity: "
            + str(loss_uniform_epochs[-1]),
            flush=True,
            file=logfile,
        )

        if args.profile:
            break

        if args.opt == "sgdslr":
            scheduler.step()

        # calculate validation loss at the end of the epoch

        t_val_start = time.time()
        with torch.no_grad():
            net.eval()
            pbar_v = tqdm.tqdm(
                val_loader,
                total=int(len(val_dataset) / args.batch_size),
                desc="Validation",
            )
            for _, batch in enumerate(pbar_v):
                batch = batch.to(args.device)  # shape (batch_size, 7, 128)
                net.optimizer.zero_grad(set_to_none=args.set_to_none)
                y_i, y_j = augmentation(args, batch)
                z_i = net(y_i, use_mask=args.mask, use_continuous_mask=args.cmask)
                z_j = net(y_j, use_mask=args.mask, use_continuous_mask=args.cmask)
                val_loss = contrastive_loss(z_i, z_j, args.temperature).to(device)
                losses_e_val.append(val_loss.detach().cpu().numpy())
                pbar_v.set_description(f"Validation loss: {val_loss:.4f}")

                writer.add_scalar(
                    "Loss/val", val_loss.item(), epoch * len(val_loader) + _
                )
            loss_e_val = np.mean(np.array(losses_e_val))
            losses_val.append(loss_e_val)
            t_val_end = time.time()
            td_val = t_val_end - t_val_start

        print(
            "epoch: "
            + str(epoch)
            + ", loss: "
            + str(round(losses[-1], 5))
            + ", val loss: "
            + str(round(losses_val[-1], 5)),
            flush=True,
            file=logfile,
        )

        # save the latest model
        torch.save(net.state_dict(), expt_dir + "model_last.pt")
        torch.save(net.optimizer.state_dict(), expt_dir + "optimizer.pt")
        if args.use_hook:
            torch.save(activations, expt_dir + "activations.pt")
        if args.use_amp:
            torch.save(scaler.state_dict(), expt_dir + "scaler.pt")

        # save the model if the validation loss is the lowest
        if losses_val[-1] < l_val_best:
            l_val_best = losses_val[-1]
            print(
                "saving out new best jetCLR model, validation loss: " + str(l_val_best),
                flush=True,
                file=logfile,
            )
            torch.save(net.state_dict(), expt_dir + "model_best.pt")

        te1 = time.time()

        if args.opt == "sgdca" or args.opt == "sgdslr":
            print("lr: " + str(scheduler._last_lr), flush=True, file=logfile)
        print(
            f"total time taken: {round( te1-te0, 1 )}s, cosine similarity: {round(td_cos_sim, 1)}s, augmentation: {round(td_aug,1)}s,, forward {round(td_forward, 1)}s, loss {round(td_loss, 1)}s, backward {round(td_backward, 1)}s, validation {round(td_val, 1)}s",
            flush=True,
            file=logfile,
        )

        # check memory stats on the gpu
        if epoch % 10 == 0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside reserved
            print(
                f"CUDA memory: total {t / np.power(1024,3)}G, reserved {r/ np.power(1024,3)}G, allocated {a/ np.power(1024,3)}G, free {f/ np.power(1024,3)}G",
                flush=True,
                file=logfile,
            )

        # check number of threads being used
        if epoch % 10 == 0:
            print(
                "num threads in use: " + str(torch.get_num_threads()),
                flush=True,
                file=logfile,
            )

        # run a short LCT
        # if epoch % 10 == 0:
        #     print("--- LCT ----", flush=True, file=logfile)
        #     # get the validation reps
        #     with torch.no_grad():
        #         net.eval()
        #         vl_reps_1 = (
        #             net.forward_batchwise(
        #                 test_dat_1.transpose(1, 2),
        #                 args.batch_size,
        #                 use_mask=args.mask,
        #                 use_continuous_mask=args.cmask,
        #             )
        #             .detach()
        #             .cpu()
        #             .numpy()
        #         )
        #         vl_reps_2 = (
        #             net.forward_batchwise(
        #                 test_dat_2.transpose(1, 2),
        #                 args.batch_size,
        #                 use_mask=args.mask,
        #                 use_continuous_mask=args.cmask,
        #             )
        #             .detach()
        #             .cpu()
        #             .numpy()
        #         )
        #         net.train()
        #     # running the LCT on each rep layer
        #     auc_list = []
        #     imtafe_list = []
        #     # loop through every representation layer
        #     for i in range(vl_reps_1.shape[1]):
        #         # just want to use the 0th rep (i.e. directly from the transformer) for now
        #         if i == 1:
        #             vl0_test = time.time()
        #             (
        #                 out_dat_vl,
        #                 out_lbs_vl,
        #                 losses_vl,
        #                 _,
        #             ) = binary_linear_classifier_test(
        #                 linear_input_size,
        #                 linear_batch_size,
        #                 linear_n_epochs,
        #                 "adam",
        #                 linear_learning_rate,
        #                 vl_reps_1[:, i, :],
        #                 test_lab_1,
        #                 vl_reps_2[:, i, :],
        #                 test_lab_2,
        #                 logfile=logfile,
        #             )
        #             auc, imtafe = get_perf_stats(out_lbs_vl, out_dat_vl)
        #             auc_list.append(auc)
        #             imtafe_list.append(imtafe)
        #             vl1_test = time.time()
        #             print(
        #                 "LCT layer "
        #                 + str(i)
        #                 + "- time taken: "
        #                 + str(np.round(vl1_test - vl0_test, 2)),
        #                 flush=True,
        #                 file=logfile,
        #             )
        #             print(
        #                 "auc: "
        #                 + str(np.round(auc, 4))
        #                 + ", imtafe: "
        #                 + str(round(imtafe, 1)),
        #                 flush=True,
        #                 file=logfile,
        #             )
        #             np.save(
        #                 expt_dir
        #                 + "lct_ep"
        #                 + str(epoch)
        #                 + "_r"
        #                 + str(i)
        #                 + "_losses.npy",
        #                 losses_vl,
        #             )
        #     auc_epochs.append(auc_list)
        #     imtafe_epochs.append(imtafe_list)
        #     print("---- --- ----", flush=True, file=logfile)

        # saving out training stats
        np.save(expt_dir + "clr_losses.npy", losses)
        np.save(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
        np.save(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))
        np.save(expt_dir + "align_loss_train.npy", loss_align_epochs)
        np.save(expt_dir + "uniform_loss_train.npy", loss_uniform_epochs)
        np.save(expt_dir + "val_losses.npy", losses_val)

    t2 = time.time()
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
        flush=True,
        file=logfile,
    )

    print(
        "JETCLR TRAINING DONE, time taken: " + str(np.round(t2 - t1, 2)),
        flush=True,
        file=logfile,
    )

    # save out results
    print("saving out data/results", flush=True, file=logfile)
    np.save(expt_dir + "clr_losses.npy", losses)
    np.save(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
    np.save(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))
    np.save(expt_dir + "align_loss_train.npy", loss_align_epochs)
    np.save(expt_dir + "uniform_loss_train.npy", loss_uniform_epochs)
    # save validation losses
    np.save(expt_dir + "val_losses.npy", losses_val)

    # save out final trained model
    print("saving out final jetCLR model", flush=True, file=logfile)
    torch.save(net.state_dict(), expt_dir + "final_model.pt")

    # print("starting the final LCT run", flush=True, file=logfile)
    # print("............................", flush=True, file=logfile)
    # with torch.no_grad():
    #     net.eval()
    #     vl_reps_1 = (
    #         net.forward_batchwise(
    #             test_dat_1.transpose(1, 2),
    #             args.batch_size,
    #             use_mask=args.mask,
    #             use_continuous_mask=args.cmask,
    #         )
    #         .detach()
    #         .cpu()
    #         .numpy()
    #     )
    #     vl_reps_2 = (
    #         net.forward_batchwise(
    #             test_dat_2.transpose(1, 2),
    #             args.batch_size,
    #             use_mask=args.mask,
    #             use_continuous_mask=args.cmask,
    #         )
    #         .detach()
    #         .cpu()
    #         .numpy()
    #     )
    #     net.train()

    # # final LCT for each rep layer
    # for i in range(vl_reps_1.shape[1]):
    #     t3 = time.time()
    #     out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test(
    #         linear_input_size,
    #         linear_batch_size,
    #         linear_n_epochs,
    #         "adam",
    #         linear_learning_rate,
    #         vl_reps_1[:, i, :],
    #         test_lab_1,
    #         vl_reps_2[:, i, :],
    #         test_lab_2,
    #         logfile=logfile,
    #     )
    #     auc, imtafe = get_perf_stats(out_lbs_f, out_dat_f)
    #     ep = 0
    #     step_size = 25
    #     for lss, val_lss in zip(losses_f[::step_size], val_losses_f):
    #         print(
    #             f"(rep layer {i}) epoch: "
    #             + str(ep)
    #             + ", loss: "
    #             + str(lss)
    #             + ", val loss: "
    #             + str(val_lss),
    #             flush=True,
    #         )
    #         ep += step_size
    #     print(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True, file=logfile)
    #     print(
    #         f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)),
    #         flush=True,
    #         file=logfile,
    #     )
    #     t4 = time.time()
    #     np.save(expt_dir + f"linear_losses_{i}.npy", losses_f)
    #     np.save(expt_dir + f"test_linear_cl_{i}.npy", out_dat_f)

    # print(
    #     "final LCT  done and output saved, time taken: " + str(np.round(t4 - t3, 2)),
    #     flush=True,
    #     file=logfile,
    # )
    print("............................", flush=True, file=logfile)

    t5 = time.time()

    print("----------------------------", flush=True, file=logfile)
    print("----------------------------", flush=True, file=logfile)
    print("----------------------------", flush=True, file=logfile)
    print(
        "ALL DONE, total time taken: " + str(np.round(t5 - t0, 2)),
        flush=True,
        file=logfile,
    )


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    # new arguments
    parser.add_argument(
        "--continue-training",
        type=int,
        action="store",
        dest="continue_training",
        default=0,
        help="continue training from a saved model",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        action="store",
        dest="base_lr",
        default=5e-5,
        help="base learning rate, to be multiplied by batch size / 128",
    )
    parser.add_argument(
        "--eps",
        type=float,
        action="store",
        dest="eps",
        default=1e-8,
        help="epsilon value for Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)",
    )
    parser.add_argument(
        "--use-hook",
        type=int,
        action="store",
        dest="use_hook",
        default=0,
        help="use hooks to identify nan or inf (https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        action="store",
        dest="max_grad_norm",
        default=0,
        help="max_norm in torch.nn.utils.clip_grad_norm_ (https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)",
    )
    parser.add_argument(
        "--use-amp",
        type=int,
        action="store",
        dest="use_amp",
        default=0,
        help="use automatic mixed precision (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)",
    )
    parser.add_argument(
        "--set-to-none",
        type=int,
        action="store",
        dest="set_to_none",
        default=0,
        help="set gradients to None in optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)",
    )
    parser.add_argument(
        "--profile",
        type=int,
        action="store",
        dest="profile",
        default=0,
        help="do profiling (only run first few batches)",
    )
    parser.add_argument(
        "--aug-device",
        type=str,
        action="store",
        dest="aug_device",
        default="cpu",
        help="device to run augmentations on (cpu or gpu)",
    )
    parser.add_argument(
        "--person",
        type=str,
        action="store",
        dest="person",
        default="Zice",
        help="Person who submittedd the job",
    )
    parser.add_argument(
        "--percent",
        type=int,
        action="store",
        dest="percent",
        help="percentage of data to use for training",
    )
    parser.add_argument(
        "--new-loss",
        type=int,
        action="store",
        dest="new_loss",
        default=0,
        help="whether to use the new loss function",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        action="store",
        dest="n_workers",
        default=2,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        action="store",
        dest="device",
        default="cuda",
        help="device to train on",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        action="store",
        dest="model_dim",
        default=1000,
        help="dimension of the transformer-encoder",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        action="store",
        dest="dim_feedforward",
        default=1000,
        help="feed forward dimension of the transformer-encoder",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        action="store",
        dest="n_layers",
        default=4,
        help="number of layers of the transformer-encoder",
    )
    parser.add_argument(
        "--n-head-layers",
        type=int,
        action="store",
        dest="n_head_layers",
        default=2,
        help="number of head layers of the transformer-encoder",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        action="store",
        dest="output_dim",
        default=1000,
        help="dimension of the output of transformer-encoder",
    )
    parser.add_argument(
        "--sbratio",
        type=float,
        action="store",
        dest="sbratio",
        default=1.0,
        help="signal to background ratio",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        action="store",
        dest="temperature",
        default=0.10,
        help="temperature hyperparameter for contrastive loss",
    )
    parser.add_argument(
        "--mask",
        type=int,
        action="store",
        help="use mask in transformer",
    )
    parser.add_argument(
        "--cmask",
        type=int,
        action="store",
        help="use continuous mask in transformer",
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        action="store",
        dest="n_epochs",
        default=300,
        help="Epochs",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=2048,
        help="batch_size",
    )
    parser.add_argument(
        "--trs",
        type=bool,
        action="store",
        dest="trs",
        default=True,
        help="do_translation",
    )
    parser.add_argument(
        "--rot",
        type=bool,
        action="store",
        dest="rot",
        default=True,
        help="do_rotation",
    )
    parser.add_argument(
        "--cf",
        type=bool,
        action="store",
        dest="cf",
        default=True,
        help="do collinear splitting",
    )
    parser.add_argument(
        "--ptd",
        type=bool,
        action="store",
        dest="ptd",
        default=True,
        help="do soft splitting (distort_jets)",
    )
    parser.add_argument(
        "--ptst",
        type=float,
        action="store",
        dest="ptst",
        default=0.1,
        help="strength param in distort_jets",
    )
    parser.add_argument(
        "--ptcm",
        type=float,
        action="store",
        dest="ptcm",
        default=0.1,
        help="pT_clip_min param in distort_jets",
    )
    parser.add_argument(
        "--trsw",
        type=float,
        action="store",
        dest="trsw",
        default=1.0,
        help="width param in translate_jets",
    )

    args = parser.parse_args()
    if args.aug_device == "cpu":
        print("Using cpu version of augmentation")
        from src.modules.jet_augs import (
            rotate_jets,
            distort_jets,
            translate_jets,
            collinear_fill_jets,
        )
    else:
        print("Using gpu version of augmentation")
        from src.modules.jet_augs_gpu import (
            rotate_jets,
            distort_jets,
            translate_jets,
            collinear_fill_jets,
        )
    main(args)
