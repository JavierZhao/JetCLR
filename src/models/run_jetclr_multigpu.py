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

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import timedelta

# Add the root directory of the project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
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

# set the number of threads that pytorch will use
torch.set_num_threads(2)


def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=60),  # Set a 60-minute timeout
    )


def cleanup():
    dist.destroy_process_group()


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
    times = [time1, time2, time3, time4, time5]
    return x_i, x_j, times


class_labels = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def plot_avg_cosine_similarities(args, avg_similarities):
    """
    Plots the average cosine similarities for each class across epochs, using a NumPy array format for avg_similarities,
    but only for epochs where the similarity is non-zero.

    :param args: Arguments containing save_plot_path.
    :param avg_similarities: NumPy array of average cosine similarities with shape [number_of_epochs, number_of_classes].
    """
    if dist.get_rank() == 0:
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


def log_info(message, file=None, flush=True):
    if dist.get_rank() == 0:
        print(message, file=file, flush=True)


def torch_save_checkpoint(state_dict, filepath):
    if dist.get_rank() == 0:
        torch.save(state_dict, filepath)


def np_save_checkpoint(filepath, arr):
    if dist.get_rank() == 0:
        np.save(filepath, arr)


def print_device_info(model):
    # Assuming 'rank' is globally accessible or passed to this function if not,
    # you might need to ensure 'rank' is defined or obtainable in this context.
    current_device = torch.cuda.current_device()

    # Retrieve the device of the first parameter of the model
    # This assumes the model has parameters and is already placed on a device.
    model_device = next(model.parameters()).device

    print(f"Current process device: {current_device}")
    print(f"Model's device: {model_device}")


def print_data_device_info(data):
    current_device = torch.cuda.current_device()
    print(f"Current process device: {current_device}")
    print(f"Data device: {data.device}")


def main(args):
    rank = args.local_rank
    torch.cuda.set_device(args.local_rank)
    # Setup for DDP: initialize process group, etc.
    setup(rank, args.world_size)
    t0 = time.time()

    args.logfile = f"/ssl-jet-vol-v3/JetCLR/logs/JetClass/{args.person}-simCLR-{args.label}-log.txt"
    args.save_plot_path = f"/ssl-jet-vol-v3/JetCLR/plots/cosine_similarity/{args.label}"
    args.n_heads = 4
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128

    # initialise logfile
    logfile = open(args.logfile, "a")
    log_info("logfile initialised", file=logfile, flush=True)
    # log_info number of workers
    log_info("number of workers: " + str(args.n_workers), file=logfile, flush=True)

    # # define the global base device
    # world_size = torch.cuda.device_count()
    # if world_size:
    #     device = torch.device("cuda:0")
    #     for i in range(world_size):
    #         log_info(
    #             f"Device {i}: {torch.cuda.get_device_name(i)}", file=logfile, flush=True
    #         )
    # else:
    #     device = torch.device("cpu")
    #     log_info("Device: CPU", file=logfile, flush=True)
    # args.device = device

    # set up results directory
    base_dir = "/ssl-jet-vol-v3/JetCLR/models/"
    expt_tag = args.label
    expt_dir = base_dir + "JetClass" + expt_tag + "/"

    # check if experiment already exists and is not empty
    if os.path.isdir(expt_dir) and os.listdir(expt_dir):
        sys.exit(
            "ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake"
        )
    else:
        # This will create the directory if it does not exist or if it is empty
        os.makedirs(expt_dir, exist_ok=True)
    log_info("experiment: " + str(args.label), file=logfile, flush=True)
    log_info(f"World size: {args.world_size}", file=logfile, flush=True)

    log_info("loading data")
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
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=rank
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=args.world_size, rank=rank
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
    log_info(f"input_dim: {input_dim}")

    # log_info data dimensions
    log_info(
        "Training data shape: " + str(train_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    log_info(
        "Validation data shape: " + str(val_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    log_info(
        "Testing data shape: " + str(test_dataset.get_dataset_shape()),
        flush=True,
        file=logfile,
    )
    log_info(
        "Testing labels shape: " + str(test_dataset.get_labels_shape()),
        flush=True,
        file=logfile,
    )

    t1 = time.time()

    log_info(
        "time taken to load data: " + str(np.round(t1 - t0, 2)) + " seconds",
        flush=True,
        file=logfile,
    )

    # set-up parameters for the LCT
    linear_input_size = args.output_dim
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 128  # we can try a larger batch size for the LCT

    log_info(
        "--- contrastive learning transformer network architecture ---",
        flush=True,
        file=logfile,
    )
    log_info("model dimension: " + str(args.model_dim), flush=True, file=logfile)
    log_info("number of heads: " + str(args.n_heads), flush=True, file=logfile)
    log_info(
        "dimension of feedforward network: " + str(args.dim_feedforward),
        flush=True,
        file=logfile,
    )
    log_info("number of layers: " + str(args.n_layers), flush=True, file=logfile)
    log_info(
        "number of head layers: " + str(args.n_head_layers), flush=True, file=logfile
    )
    log_info("optimiser: " + str(args.opt), flush=True, file=logfile)
    log_info(f"use mask: {True if args.mask else False}", flush=True, file=logfile)
    log_info(
        f"use continuous mask: {True if args.cmask else False}",
        flush=True,
        file=logfile,
    )
    log_info("\n--- hyper-parameters ---", flush=True, file=logfile)
    log_info("learning rate: " + str(args.learning_rate), flush=True, file=logfile)
    log_info("batch size: " + str(args.batch_size), flush=True, file=logfile)
    log_info("temperature: " + str(args.temperature), flush=True, file=logfile)
    log_info("\n--- symmetries/augmentations ---", flush=True, file=logfile)
    log_info("rotations: " + str(args.rot), flush=True, file=logfile)
    log_info("low pT smearing: " + str(args.ptd), flush=True, file=logfile)
    log_info("pT smearing clip parameter: " + str(args.ptcm), flush=True, file=logfile)
    log_info("translations: " + str(args.trs), flush=True, file=logfile)
    log_info("translations width: " + str(args.trsw), flush=True, file=logfile)
    log_info(
        "Number of input features per particle: " + str(input_dim),
        flush=True,
        file=logfile,
    )
    log_info("---------------", flush=True, file=logfile)

    # initialise the network
    log_info("\ninitialising the network", flush=True, file=logfile)
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

    # Move the model to the correct device
    device = torch.device(f"cuda:{rank}")
    args.device = device
    net.to(device)
    net = DDP(net, device_ids=[rank])
    log_info("\nmodel initialised and sent to devices", flush=True, file=logfile)

    # set up the optimiser
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

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
    log_info(
        "starting training loop, running for " + str(args.n_epochs) + " epochs",
        flush=True,
        file=logfile,
    )
    log_info("---------------", flush=True, file=logfile)

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
        log_info(
            "number of iterations per epoch: " + str(iters), flush=True, file=logfile
        )

        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.n_workers,
        #     pin_memory=True,
        #     prefetch_factor=2,
        #     sampler=train_sampler,
        # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=2,
        sampler=val_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=test_sampler,
    )
    num_classes = 10  # Assuming classes are labeled 0 through 9
    average_similarities = np.zeros((args.n_epochs, num_classes))

    l_val_best = 1000000  # initialise the best validation loss
    # the loop
    for epoch in range(args.n_epochs):
        log_info("epoch: " + str(epoch), flush=True, file=logfile)
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # print_device_info(net)
        # initialise timing stats
        te0 = time.time()

        # initialise lists to store batch stats
        loss_align_e = []
        loss_uniform_e = []
        losses_e = []
        losses_e_val = []

        # initialise timing stats
        td1 = 0
        td2 = 0
        td3 = 0
        td4 = 0
        td5 = 0
        td6 = 0
        td7 = 0
        td8 = 0

        # calculate the average cosine similarity for each class
        # initialize the arrays to store the average cosine similarities
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

                    avg_similarities[label] += sim_value
                    counts[label] += 1

            average_similarities[epoch, :] = np.divide(
                avg_similarities, counts, where=counts != 0
            )
        # save average similarities
        np_save_checkpoint(expt_dir + "average_similarities.npy", average_similarities)

        plot_avg_cosine_similarities(args, average_similarities)

        # print(f"len(train_dataset): {len(train_dataset)}")
        # for i, data in enumerate(train_dataset):
        #     print(f"Data sample {i}: {data.shape}")
        #     if i == 10:
        #         break

        # for idx in train_sampler:
        #     log_info(idx, file=logfile, flush=True)
        #     # Check if indices are being generated

        # first_batch = next(iter(train_loader))
        # print(f"first_batch shape: {first_batch.shape}")

        # iterate over the training loader to make sure it works
        # for i, batch in enumerate(train_loader):
        #     log_info("batch: " + str(i), flush=True, file=logfile)
        #     # batch = batch.to(args.device)
        #     # print_data_device_info(batch)
        #     if i == 10:
        #         break

        # the inner loop goes through the dataset batch by batch
        # augmentations of the jets are done on the fly

        net.train()
        total_train = int(len(train_dataset) / (args.batch_size * args.world_size))
        pbar_t = tqdm.tqdm(
            train_loader,
            total=total_train,
            desc="Training",
        )
        log_info("total batches: " + str(total_train), flush=True, file=logfile)
        for i, batch in enumerate(pbar_t):
            # log_info("batch: " + str(i), flush=True, file=logfile)
            batch = batch.to(args.device)  # shape (batch_size, 7, 128)
            # print_data_device_info(batch)
            optimizer.zero_grad()
            x_i, x_j, times = augmentation(args, batch)
            time1, time2, time3, time4, time5 = times
            time6 = time.time()
            z_i = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask)
            z_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
            time7 = time.time()
            # calculate the alignment and uniformity loss for each batch
            loss_align = align_loss(z_i, z_j)
            loss_uniform_zi = uniform_loss(z_i)
            loss_uniform_zj = uniform_loss(z_j)
            loss_align_e.append(loss_align.detach().cpu().numpy())
            loss_uniform_e.append(
                (
                    loss_uniform_zi.detach().cpu().numpy()
                    + loss_uniform_zj.detach().cpu().numpy()
                )
                / 2
            )
            time8 = time.time()

            # compute the loss, back-propagate, and update scheduler if required
            loss = contrastive_loss(z_i, z_j, args.temperature).to(device)
            loss.backward()
            optimizer.step()
            if args.opt == "sgdca":
                scheduler.step(epoch + i / iters)
            losses_e.append(loss.detach().cpu().numpy())
            time9 = time.time()
            pbar_t.set_description(f"Training loss: {loss:.4f}")

            # update timiing stats
            td1 += time2 - time1
            td2 += time3 - time2
            td3 += time4 - time3
            td4 += time5 - time4
            td5 += time6 - time5
            td6 += time7 - time6
            td7 += time8 - time7
            td8 += time9 - time8

        loss_e = np.mean(np.array(losses_e))
        losses.append(loss_e)

        if args.opt == "sgdslr":
            scheduler.step()

        te1 = time.time()

        # calculate validation loss at the end of the epoch

        with torch.no_grad():
            net.eval()
            pbar_v = tqdm.tqdm(
                val_loader,
                total=int(len(val_dataset) / args.batch_size),
                desc="Validation",
            )
            for _, batch in enumerate(pbar_v):
                batch = batch.to(args.device)  # shape (batch_size, 7, 128)
                optimizer.zero_grad()
                y_i, y_j, times = augmentation(args, batch)
                time1, time2, time3, time4, time5 = times
                z_i = net(y_i, use_mask=args.mask, use_continuous_mask=args.cmask)
                z_j = net(y_j, use_mask=args.mask, use_continuous_mask=args.cmask)
                val_loss = contrastive_loss(z_i, z_j, args.temperature).to(device)
                losses_e_val.append(val_loss.detach().cpu().numpy())
                pbar_v.set_description(f"Validation loss: {val_loss:.4f}")
            loss_e_val = np.mean(np.array(losses_e_val))
            losses_val.append(loss_e_val)

        log_info(
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
        torch_save_checkpoint(net.state_dict(), expt_dir + "model_last.pt")

        # save the model if the validation loss is the lowest
        if losses_val[-1] < l_val_best:
            l_val_best = losses_val[-1]
            log_info(
                "saving out new best jetCLR model, validation loss: " + str(l_val_best),
                flush=True,
                file=logfile,
            )
            torch_save_checkpoint(net.state_dict(), expt_dir + "model_best.pt")

        # summarise alignment and uniformity stats
        loss_align_epochs.append(np.mean(np.array(loss_align_e)))
        loss_uniform_epochs.append(np.mean(np.array(loss_uniform_e)))
        log_info(
            "alignment: "
            + str(loss_align_epochs[-1])
            + ", uniformity: "
            + str(loss_uniform_epochs[-1]),
            flush=True,
            file=logfile,
        )

        if args.opt == "sgdca" or args.opt == "sgdslr":
            log_info("lr: " + str(scheduler._last_lr), flush=True, file=logfile)
        log_info(
            f"total time taken: {round( te1-te0, 1 )}s, augmentation: {round(td1+td2+td3+td4+td5,1)}s, forward {round(td6, 1)}s, backward {round(td8, 1)}s, other {round(te1-te0-(td1+td2+td3+td4+td6+td7+td8), 2)}s",
            flush=True,
            file=logfile,
        )

        # check memory stats on the gpu
        if epoch % 10 == 0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside reserved
            log_info(
                f"CUDA memory: total {t / np.power(1024,3)}G, reserved {r/ np.power(1024,3)}G, allocated {a/ np.power(1024,3)}G, free {f/ np.power(1024,3)}G",
                flush=True,
                file=logfile,
            )

        # check number of threads being used
        if epoch % 10 == 0:
            log_info(
                "num threads in use: " + str(torch.get_num_threads()),
                flush=True,
                file=logfile,
            )

        # run a short LCT
        # if epoch % 10 == 0:
        #     log_info("--- LCT ----", flush=True, file=logfile)
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
        #             log_info(
        #                 "LCT layer "
        #                 + str(i)
        #                 + "- time taken: "
        #                 + str(np.round(vl1_test - vl0_test, 2)),
        #                 flush=True,
        #                 file=logfile,
        #             )
        #             log_info(
        #                 "auc: "
        #                 + str(np.round(auc, 4))
        #                 + ", imtafe: "
        #                 + str(round(imtafe, 1)),
        #                 flush=True,
        #                 file=logfile,
        #             )
        #             np_save_checkpoint(
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
        #     log_info("---- --- ----", flush=True, file=logfile)

        # saving out training stats
        np_save_checkpoint(expt_dir + "clr_losses.npy", losses)
        np_save_checkpoint(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
        np_save_checkpoint(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))
        np_save_checkpoint(expt_dir + "align_loss_train.npy", loss_align_epochs)
        np_save_checkpoint(expt_dir + "uniform_loss_train.npy", loss_uniform_epochs)
        np_save_checkpoint(expt_dir + "val_losses.npy", losses_val)

    t2 = time.time()

    log_info(
        "JETCLR TRAINING DONE, time taken: " + str(np.round(t2 - t1, 2)),
        flush=True,
        file=logfile,
    )

    # save out results
    log_info("saving out data/results", flush=True, file=logfile)
    np_save_checkpoint(expt_dir + "clr_losses.npy", losses)
    np_save_checkpoint(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
    np_save_checkpoint(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))
    np_save_checkpoint(expt_dir + "align_loss_train.npy", loss_align_epochs)
    np_save_checkpoint(expt_dir + "uniform_loss_train.npy", loss_uniform_epochs)
    # save validation losses
    np_save_checkpoint(expt_dir + "val_losses.npy", losses_val)

    # save out final trained model
    log_info("saving out final jetCLR model", flush=True, file=logfile)
    torch_save_checkpoint(net.state_dict(), expt_dir + "final_model.pt")

    # log_info("starting the final LCT run", flush=True, file=logfile)
    # log_info("............................", flush=True, file=logfile)
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
    #         log_info(
    #             f"(rep layer {i}) epoch: "
    #             + str(ep)
    #             + ", loss: "
    #             + str(lss)
    #             + ", val loss: "
    #             + str(val_lss),
    #             flush=True,
    #         )
    #         ep += step_size
    #     log_info(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True, file=logfile)
    #     log_info(
    #         f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)),
    #         flush=True,
    #         file=logfile,
    #     )
    #     t4 = time.time()
    #     np_save_checkpoint(expt_dir + f"linear_losses_{i}.npy", losses_f)
    #     np_save_checkpoint(expt_dir + f"test_linear_cl_{i}.npy", out_dat_f)

    # log_info(
    #     "final LCT  done and output saved, time taken: " + str(np.round(t4 - t3, 2)),
    #     flush=True,
    #     file=logfile,
    # )
    log_info("............................", flush=True, file=logfile)

    t5 = time.time()

    log_info("----------------------------", flush=True, file=logfile)
    log_info("----------------------------", flush=True, file=logfile)
    log_info("----------------------------", flush=True, file=logfile)
    log_info(
        "ALL DONE, total time taken: " + str(np.round(t5 - t0, 2)),
        flush=True,
        file=logfile,
    )
    cleanup()


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    # new arguments
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
    parser.add_argument(
        "--local-rank",
        type=int,
        action="store",
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )

    args = parser.parse_args()
    args.world_size = torch.cuda.device_count()
    main(args)
