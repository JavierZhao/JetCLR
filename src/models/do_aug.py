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
from torch.utils.data import DataLoader

# load custom modules required for jetCLR training
# from src.modules.jet_augs import (
#     rotate_jets,
#     distort_jets,
#     translate_jets,
#     collinear_fill_jets,
# )
from src.modules.transformer import Transformer
from src.modules.losses import contrastive_loss, align_loss, uniform_loss
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
    
    # torch.cuda.synchronize()  # Ensure all operations are completed

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
    ).transpose(1, 2)  # (batch_size, 6, n_constit)

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
    ).transpose(1, 2)  # (batch_size, 6, n_constit)
    return x_i, x_j

class_labels = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]



def main(args):
    t0 = time.time()

    args.logfile = f"/ssl-jet-vol-v3/JetCLR/logs/JetClass/{args.person}-simCLR-{args.label}-log.txt"
    args.save_plot_path = f"/ssl-jet-vol-v3/JetCLR/plots/cosine_similarity/{args.label}"
    args.n_heads = 4
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128

    # initialise logfile
    logfile = open(args.logfile, "a")
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
    if os.path.isdir(expt_dir) and os.listdir(expt_dir):
        sys.exit(
            "ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake"
        )
    else:
        # This will create the directory if it does not exist or if it is empty
        os.makedirs(expt_dir, exist_ok=True)
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


    input_dim = train_dataset.get_sample_shape()[0]

    t1 = time.time()

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
    )

    # send network to device
    net.to(device)

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
    # the loop
    for epoch in range(args.n_epochs):
        if epoch == 1:
            break
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

        t_cos_sim_end = time.time()
        td_cos_sim = t_cos_sim_end - t_cos_sim_start
        # print(f"time to do augmentations without collinear filling on {args.aug_device}: {round(td_cos_sim, 1)}s")
        print(f"time to do augmentations on {args.aug_device}: {round(td_cos_sim, 1)}s")


       


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    # new arguments
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
