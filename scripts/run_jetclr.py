#!/bin/env python3.7

# load standard python modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import argparse
import copy

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules required for jetCLR training
from .modules.jet_augs import (
    rotate_jets,
    distort_jets,
    rescale_pts,
    crop_jets,
    translate_jets,
    collinear_fill_jets,
)
from .modules.transformer import Transformer
from .modules.losses import contrastive_loss, align_loss, uniform_loss
from .modules.perf_eval import get_perf_stats, linear_classifier_test

# import args from extargs.py file
# import extargs as args

# set the number of threads that pytorch will use
torch.set_num_threads(2)


# load data
def load_data(dataset_path, flag, n_files=-1):
    # make another variable that combines flag and subdirectory such as 3_features_raw
    path_id = f"{flag}-"
    if args.full_kinematics:
        data_files = glob.glob(f"{dataset_path}/{flag}/processed/7_features_raw/data/*")
        path_id += "7_features_raw"
    elif args.raw_3:
        data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features_raw/data/*")
        path_id += "3_features_raw"
    else:
        data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/data/*")
        path_id += "3_features_relative"

    data = []
    for i, _ in enumerate(data_files):
        if args.full_kinematics:
            data.append(
                np.load(
                    f"{dataset_path}/{flag}/processed/7_features_raw/data/data_{i}.npy"
                )
            )
        elif args.raw_3:
            data.append(
                torch.load(
                    f"{dataset_path}/{flag}/processed/3_features_raw/data/data_{i}.pt"
                ).numpy()
            )
        else:
            data.append(
                torch.load(
                    f"{dataset_path}/{flag}/processed/3_features/data/data_{i}.pt"
                ).numpy()
            )

        print(f"--- loaded file {i} from `{path_id}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def load_labels(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/7_features_raw/labels/*")

    data = []
    for i, file in enumerate(data_files):
        data.append(
            np.load(
                f"{dataset_path}/{flag}/processed/7_features_raw/labels/labels_{i}.npy"
            )
        )
        print(f"--- loaded label file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def augmentation(args, x_i):
    if args.full_kinematics:
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
    if not args.full_kinematics:
        x_i = rescale_pts(x_i)
        x_j = rescale_pts(x_j)
    if args.full_kinematics:
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
            [
                eta_i,
                phi_i,
                pT_log_i,
                E_log_i,
                pt_rel_log_i,
                E_rel_log_i,
                deltaR_i,
            ],
            1,
        )  # (batch_size, 7, n_constit)
        x_j = np.stack(
            [
                eta_j,
                phi_j,
                pT_log_j,
                E_log_j,
                pt_rel_log_j,
                E_rel_log_j,
                deltaR_j,
            ],
            1,
        )  # (batch_size, 7, n_constit)
    x_i = torch.Tensor(x_i).transpose(1, 2).to(args.device)
    x_j = torch.Tensor(x_j).transpose(1, 2).to(args.device)
    times = [time1, time2, time3, time4, time5]
    return x_i, x_j, times


def main(args):
    t0 = time.time()
    print(f"full_kinematics: {args.full_kinematics}")
    print(f"raw_3: {args.raw_3}")
    print(f"use mask: {args.mask}")
    print(f"use continuous mask: {args.cmask}")
    args.logfile = f"/ssl-jet-vol-v2/JetCLR/logs/zz-simCLR-{args.label}-log.txt"
    args.nconstit = 50
    args.n_heads = 4
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128

    # initialise logfile
    logfile = open(args.logfile, "a")
    print("logfile initialised", file=logfile, flush=True)

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
    base_dir = "/ssl-jet-vol-v2/JetCLR/models/"
    expt_tag = args.label
    expt_dir = base_dir + "experiments/" + expt_tag + "/"

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
    data = load_data("/ssl-jet-vol-v2/toptagging", "train", args.num_files)
    data_val = load_data("/ssl-jet-vol-v2/toptagging", "val", 1)
    labels = load_labels("/ssl-jet-vol-v2/toptagging", "train", args.num_files)
    labels_val = load_labels("/ssl-jet-vol-v2/toptagging", "val", 1)
    tr_dat_in = np.concatenate(data, axis=0)  # Concatenate along the first axis
    val_dat_in = np.concatenate(data_val, axis=0)
    # reduce validation data
    val_dat_in = val_dat_in[0:10000]
    tr_lab_in = np.concatenate(labels, axis=0)
    val_lab_in = np.concatenate(labels_val, axis=0)
    val_lab_in = val_lab_in[0:10000]

    # creating the training dataset
    print("shuffling data and doing the S/B split", flush=True, file=logfile)
    tr_bkg_dat = tr_dat_in[tr_lab_in == 0].copy()
    tr_sig_dat = tr_dat_in[tr_lab_in == 1].copy()
    nbkg_tr = int(tr_bkg_dat.shape[0])
    nsig_tr = int(args.sbratio * nbkg_tr)
    list_tr_dat = list(tr_bkg_dat[0:nbkg_tr]) + list(tr_sig_dat[0:nsig_tr])
    list_tr_lab = [0 for i in range(nbkg_tr)] + [1 for i in range(nsig_tr)]
    ldz_tr = list(zip(list_tr_dat, list_tr_lab))
    random.shuffle(ldz_tr)
    tr_dat, tr_lab = zip(*ldz_tr)
    tr_dat = np.array(tr_dat)
    tr_lab = np.array(tr_lab)

    # do the same with the validation dataset
    print(
        "shuffling data and doing the S/B split for the validation dataset",
        flush=True,
        file=logfile,
    )
    vl_bkg_dat = val_dat_in[val_lab_in == 0].copy()
    vl_sig_dat = val_dat_in[val_lab_in == 1].copy()
    nbkg_vl = int(vl_bkg_dat.shape[0])
    nsig_vl = int(args.sbratio * nbkg_vl)
    list_test_dat = list(vl_bkg_dat[0:nbkg_vl]) + list(vl_sig_dat[0:nsig_vl])
    list_test_lab = [0 for i in range(nbkg_vl)] + [1 for i in range(nsig_vl)]
    ldz_test = list(zip(list_test_dat, list_test_lab))
    random.shuffle(ldz_test)
    vl_dat, vl_lab = zip(*ldz_test)
    vl_dat = np.array(vl_dat)
    vl_lab = np.array(vl_lab)

    # take out the delta_R feature
    # if args.full_kinematics:
    #     tr_dat = tr_dat[:, 0:6, :]
    #     val_dat_in = val_dat_in[:, 0:6, :]
    # input dim to the transformer -> (pt,eta,phi)
    input_dim = tr_dat.shape[1]
    print(f"input_dim: {input_dim}")

    # create two testing sets:
    # one for training the linear classifier test (LCT)
    # and one for testing on it
    # we will do this just with tr_dat_in, but shuffled and split 50/50
    # this should be fine because the jetCLR training doesn't use labels
    # we want the LCT to use S/B=1 all the time
    list_test_dat = list(tr_dat_in.copy())
    list_test_lab = list(tr_lab_in.copy())
    ldz_test = list(zip(list_test_dat, list_test_lab))
    random.shuffle(ldz_test)
    test_dat, test_lab = zip(*ldz_test)
    test_dat = np.array(test_dat)
    test_lab = np.array(test_lab)
    test_len = test_dat.shape[0]
    test_split_len = int(test_len / 2)
    test_dat_1 = test_dat[0:test_split_len]
    test_lab_1 = test_lab[0:test_split_len]
    test_dat_2 = test_dat[-test_split_len:]
    test_lab_2 = test_lab[-test_split_len:]

    # cropping all jets to a fixed number of consituents
    # tr_dat = crop_jets(tr_dat, args.nconstit)
    # test_dat_1 = crop_jets(test_dat_1, args.nconstit)
    # test_dat_2 = crop_jets(test_dat_2, args.nconstit)

    # reducing the testing data for consistency
    test_cut = 50000  # 50k jets
    test_dat_1 = test_dat_1[0:test_cut]
    test_lab_1 = test_lab_1[0:test_cut]
    test_dat_2 = test_dat_2[0:test_cut]
    test_lab_2 = test_lab_2[0:test_cut]

    # print data dimensions
    print("training data shape: " + str(tr_dat.shape), flush=True, file=logfile)
    print("Testing-1 data shape: " + str(test_dat_1.shape), flush=True, file=logfile)
    print("Testing-2 data shape: " + str(test_dat_2.shape), flush=True, file=logfile)
    print("validation data shape: " + str(vl_dat.shape), flush=True, file=logfile)
    print("training labels shape: " + str(tr_lab.shape), flush=True, file=logfile)
    print("Testing-1 labels shape: " + str(test_lab_1.shape), flush=True, file=logfile)
    print("Testing-2 labels shape: " + str(test_lab_2.shape), flush=True, file=logfile)
    print("validation labels shape: " + str(vl_lab.shape), flush=True, file=logfile)

    t1 = time.time()

    # re-scale test data, for the training data this will be done on the fly due to the augmentations
    if not args.full_kinematics:
        test_dat_1 = rescale_pts(test_dat_1)
        test_dat_2 = rescale_pts(test_dat_2)

    print(
        "time taken to load and preprocess data: "
        + str(np.round(t1 - t0, 2))
        + " seconds",
        flush=True,
        file=logfile,
    )

    # set-up parameters for the LCT
    linear_input_size = args.output_dim
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 128

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
    print("mask: " + str(args.mask), flush=True, file=logfile)
    print("continuous mask: " + str(args.cmask), flush=True, file=logfile)
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
    if args.full_kinematics:
        print("Number of input features per particle: 7", flush=True, file=logfile)
    else:
        print("Number of input features per particle: 3", flush=True, file=logfile)
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
        log=args.full_kinematics,
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
        iters = int(tr_dat.shape[0] / args.batch_size)
        print("number of iterations per epoch: " + str(iters), flush=True, file=logfile)

    # the loop
    for epoch in range(args.n_epochs):
        # re-batch the data on each epoch
        indices_list = torch.split(torch.randperm(tr_dat.shape[0]), args.batch_size)
        indices_list_val = torch.split(torch.randperm(vl_dat.shape[0]), args.batch_size)

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

        # the inner loop goes through the dataset batch by batch
        # augmentations of the jets are done on the fly
        for i, indices in enumerate(indices_list):
            net.optimizer.zero_grad()
            x_i = tr_dat[indices, :, :]
            x_i, x_j, times = augmentation(args, x_i)
            time1, time2, time3, time4, time5 = times
            time6 = time.time()
            z_i = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask)
            z_j = net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
            time7 = time.time()

            # calculate the alignment and uniformity loss for each batch
            if epoch % 10 == 0:
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
            net.optimizer.step()
            if args.opt == "sgdca":
                scheduler.step(epoch + i / iters)
            losses_e.append(loss.detach().cpu().numpy())
            time9 = time.time()

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

            # do augmentations on the fly for the validation data
            for i, indices in enumerate(indices_list_val):
                net.optimizer.zero_grad()
                y_i = vl_dat[indices, :, :]
                y_i, y_j, times = augmentation(args, y_i)
                z_i = net(y_i, use_mask=args.mask, use_continuous_mask=args.cmask)
                z_j = net(y_j, use_mask=args.mask, use_continuous_mask=args.cmask)
                val_loss = contrastive_loss(z_i, z_j, args.temperature).to(device)
                losses_e_val.append(val_loss.detach().cpu().numpy())
            net.train()
            loss_e_val = np.mean(np.array(losses_e_val))
            losses_val.append(loss_e_val)

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

        if args.opt == "sgdca" or args.opt == "sgdslr":
            print("lr: " + str(scheduler._last_lr), flush=True, file=logfile)
        print(
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
            print(
                f"CUDA memory: total {t / np.power(1024,3)}G, reserved {r/ np.power(1024,3)}G, allocated {a/ np.power(1024,3)}G, free {f/ np.power(1024,3)}G",
                flush=True,
                file=logfile,
            )

        # summarise alignment and uniformity stats
        if epoch % 10 == 0:
            loss_align_epochs.append(np.mean(np.array(loss_align_e)))
            loss_uniform_epochs.append(np.mean(np.array(loss_uniform_e)))
            print(
                "alignment: "
                + str(loss_align_epochs[-1])
                + ", uniformity: "
                + str(loss_uniform_epochs[-1]),
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

        # saving the model
        if epoch % 10 == 0:
            print("saving out jetCLR model", flush=True, file=logfile)
            tms0 = time.time()
            torch.save(net.state_dict(), expt_dir + "model_ep" + str(epoch) + ".pt")
            tms1 = time.time()
            print(
                f"time taken to save model: {round( tms1-tms0, 1 )}s",
                flush=True,
                file=logfile,
            )

        # run a short LCT
        if epoch % 10 == 0:
            print("--- LCT ----", flush=True, file=logfile)
            # if args.trs:
            #     test_dat_1 = translate_jets( test_dat_1, width=args.trsw )
            #     test_dat_2 = translate_jets( test_dat_2, width=args.trsw )
            # get the validation reps
            with torch.no_grad():
                net.eval()
                # vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( test_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
                # vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( test_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
                vl_reps_1 = (
                    net.forward_batchwise(
                        torch.Tensor(test_dat_1).transpose(1, 2),
                        args.batch_size,
                        use_mask=args.mask,
                        use_continuous_mask=args.cmask,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                vl_reps_2 = (
                    net.forward_batchwise(
                        torch.Tensor(test_dat_2).transpose(1, 2),
                        args.batch_size,
                        use_mask=args.mask,
                        use_continuous_mask=args.cmask,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                net.train()
            # running the LCT on each rep layer
            auc_list = []
            imtafe_list = []
            # loop through every representation layer
            for i in range(vl_reps_1.shape[1]):
                # just want to use the 0th rep (i.e. directly from the transformer) for now
                if i == 1:
                    vl0_test = time.time()
                    (
                        out_dat_vl,
                        out_lbs_vl,
                        losses_vl,
                        _,
                    ) = linear_classifier_test(
                        linear_input_size,
                        linear_batch_size,
                        linear_n_epochs,
                        "adam",
                        linear_learning_rate,
                        vl_reps_1[:, i, :],
                        np.expand_dims(test_lab_1, axis=1),
                        vl_reps_2[:, i, :],
                        np.expand_dims(test_lab_2, axis=1),
                        logfile=logfile,
                    )
                    auc, imtafe = get_perf_stats(out_lbs_vl, out_dat_vl)
                    auc_list.append(auc)
                    imtafe_list.append(imtafe)
                    vl1_test = time.time()
                    print(
                        "LCT layer "
                        + str(i)
                        + "- time taken: "
                        + str(np.round(vl1_test - vl0_test, 2)),
                        flush=True,
                        file=logfile,
                    )
                    print(
                        "auc: "
                        + str(np.round(auc, 4))
                        + ", imtafe: "
                        + str(round(imtafe, 1)),
                        flush=True,
                        file=logfile,
                    )
                    np.save(
                        expt_dir
                        + "lct_ep"
                        + str(epoch)
                        + "_r"
                        + str(i)
                        + "_losses.npy",
                        losses_vl,
                    )
            auc_epochs.append(auc_list)
            imtafe_epochs.append(imtafe_list)
            print("---- --- ----", flush=True, file=logfile)

        # saving out training stats
        if epoch % 10 == 0:
            print("saving out data/results", flush=True, file=logfile)
            tds0 = time.time()
            np.save(expt_dir + "clr_losses.npy", losses)
            np.save(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
            np.save(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))
            np.save(expt_dir + "align_loss_train.npy", loss_align_epochs)
            np.save(expt_dir + "uniform_loss_train.npy", loss_uniform_epochs)
            tds1 = time.time()
            print(
                f"time taken to save data: {round( tds1-tds0, 1 )}s",
                flush=True,
                file=logfile,
            )

    t2 = time.time()

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

    # save out final trained model
    print("saving out final jetCLR model", flush=True, file=logfile)
    torch.save(net.state_dict(), expt_dir + "final_model.pt")

    print("starting the final LCT run", flush=True, file=logfile)

    # evaluate the network on the testing data, applying some augmentations first if it's required
    # if args.trs:
    #     test_dat_1 = translate_jets( test_dat_1, width=args.trsw )
    #     test_dat_2 = translate_jets( test_dat_2, width=args.trsw )
    with torch.no_grad():
        net.eval()
        # vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( test_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
        # vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( test_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
        vl_reps_1 = (
            net.forward_batchwise(
                torch.Tensor(test_dat_1).transpose(1, 2),
                args.batch_size,
                use_mask=args.mask,
                use_continuous_mask=args.cmask,
            )
            .detach()
            .cpu()
            .numpy()
        )
        vl_reps_2 = (
            net.forward_batchwise(
                torch.Tensor(test_dat_2).transpose(1, 2),
                args.batch_size,
                use_mask=args.mask,
                use_continuous_mask=args.cmask,
            )
            .detach()
            .cpu()
            .numpy()
        )
        net.train()

    # final LCT for each rep layer
    for i in range(vl_reps_1.shape[1]):
        t3 = time.time()
        out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test(
            linear_input_size,
            linear_batch_size,
            linear_n_epochs,
            "adam",
            linear_learning_rate,
            vl_reps_1[:, i, :],
            np.expand_dims(test_lab_1, axis=1),
            vl_reps_2[:, i, :],
            np.expand_dims(test_lab_2, axis=1),
            logfile=logfile,
        )
        auc, imtafe = get_perf_stats(out_lbs_f, out_dat_f)
        ep = 0
        step_size = 25
        for lss, val_lss in zip(losses_f[::step_size], val_losses_f):
            print(
                f"(rep layer {i}) epoch: "
                + str(ep)
                + ", loss: "
                + str(lss)
                + ", val loss: "
                + str(val_lss),
                flush=True,
            )
            ep += step_size
        print(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True, file=logfile)
        print(
            f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)),
            flush=True,
            file=logfile,
        )
        t4 = time.time()
        np.save(expt_dir + f"linear_losses_{i}.npy", losses_f)
        np.save(expt_dir + f"test_linear_cl_{i}.npy", out_dat_f)

    print(
        "final LCT  done and output saved, time taken: " + str(np.round(t4 - t3, 2)),
        flush=True,
        file=logfile,
    )
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
    parser.add_argument(
        "--device",
        type=str,
        action="store",
        dest="device",
        default="cuda",
        help="device to train on",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v2/toptagging/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        action="store",
        dest="num_files",
        default=12,
        help="number of files for training",
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
        "--shared",
        type=bool,
        action="store",
        default=True,
        help="share parameters of backbone",
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
        "--nconstit",
        type=int,
        action="store",
        dest="nconstit",
        default=50,
        help="number of constituents per jet",
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
        "--full-kinematics",
        type=int,
        action="store",
        dest="full_kinematics",
        help="use the full 7 kinematic features instead of just 3",
    )
    parser.add_argument(
        "--raw-3",
        type=int,
        action="store",
        dest="raw_3",
        help="use the 3 raw features",
    )

    args = parser.parse_args()
    main(args)
