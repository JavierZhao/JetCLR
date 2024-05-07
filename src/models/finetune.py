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
from pathlib import Path
import math

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn import metrics


# load custom modules required for jetCLR training
from src.modules.jet_augs import (
    rotate_jets,
    distort_jets,
    translate_jets,
    collinear_fill_jets,
    rescale_pts,
)
from src.modules.transformer import Transformer
from src.modules.losses import contrastive_loss, align_loss, uniform_loss
from src.modules.perf_eval import get_perf_stats, linear_classifier_test

# set the number of threads that pytorch will use
torch.set_num_threads(2)


# the MLP projector
def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


# load data
def load_data(dataset_path, flag, n_files=-1):
    # make another variable that combines flag and subdirectory such as 3_features_raw
    path_id = f"{flag}-"
    if args.full_kinematics:
        data_files = glob.glob(f"{dataset_path}/{flag}/processed/7_features_raw/data/*")
        path_id += "7_features_raw"
    elif args.six_features:
        data_files = glob.glob(f"{dataset_path}/{flag}/processed/6_features_raw/data/*")
        path_id += "6_features"
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
        elif args.six_features:
            data.append(
                np.load(
                    f"{dataset_path}/{flag}/processed/6_features_raw/data/data_{i}.npy"
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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_perf_stats(labels, measures):
    measures = np.nan_to_num(measures)
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, thresholds = metrics.roc_curve(labels, measures)
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= 0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= 0.5]
    try:
        imtafe = np.nan_to_num(
            1 / fpr2[list(tpr2).index(find_nearest(list(tpr2), 0.5))]
        )
    except:
        imtafe = 1
    return auc, imtafe


def main(args):
    t0 = time.time()
    print(f"full_kinematics: {args.full_kinematics}")
    print(f"six_features: {args.six_features}")
    print(f"raw_3: {args.raw_3}")
    print(f"use mask: {args.mask}")
    print(f"use continuous mask: {args.cmask}")
    # set up results directory
    base_dir = "/ssl-jet-vol-v3/JetCLR/models/"
    expt_tag = (
        f"trial-{args.trial}-{args.ep}-{math.log10(args.num_samples)}-{args.label}"
    )
    if not args.finetune:
        expt_tag += "-fixed"
    if args.group_tag:
        expt_dir = base_dir + "finetuning/" + args.group_tag + "/" + expt_tag + "/"
    else:
        expt_dir = base_dir + "finetuning/" + expt_tag + "/"

    if args.group_tag:
        args.logfile = (
            f"/ssl-jet-vol-v3/JetCLR/logs/finetuning/{args.group_tag}/{expt_tag}.txt"
        )
    else:
        args.logfile = f"/ssl-jet-vol-v3/JetCLR/logs/finetuning/{expt_tag}.txt"
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

    # check if experiment already exists and is not empty

    if os.path.isdir(expt_dir) and os.listdir(expt_dir):
        sys.exit(
            "ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake"
        )
    else:
        # This will create the directory if it does not exist or if it is empty
        os.makedirs(expt_dir, exist_ok=True)
    print("experiment: " + str(expt_tag), file=logfile, flush=True)

    # print purpose of experiment
    if "from_scratch" in args.label:
        print("training from scratch", file=logfile, flush=True)
    print(f"finetune: {args.finetune}", file=logfile, flush=True)

    print("loading data")
    args.num_files = args.num_samples // 100000 + 1
    data = load_data("/ssl-jet-vol-v3/toptagging", "train", args.num_files)
    data_val = load_data("/ssl-jet-vol-v3/toptagging", "val", 1)
    labels = load_labels("/ssl-jet-vol-v3/toptagging", "train", args.num_files)
    labels_val = load_labels("/ssl-jet-vol-v3/toptagging", "val", 1)
    tr_dat_in = np.concatenate(data, axis=0)  # Concatenate along the first axis
    val_dat_in = np.concatenate(data_val, axis=0)
    tr_dat_in = tr_dat_in[0 : args.num_samples]
    # reduce validation data
    val_dat_in = val_dat_in[0:10000]
    tr_lab_in = np.concatenate(labels, axis=0)
    tr_lab_in = tr_lab_in[0 : args.num_samples]
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

    # if not args.full_kinematics:
    #     test_dat_1 = rescale_pts(test_dat_1)
    #     test_dat_2 = rescale_pts(test_dat_2)

    print(
        "time taken to load and preprocess data: "
        + str(np.round(t1 - t0, 2))
        + " seconds",
        flush=True,
        file=logfile,
    )

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
    print(
        f"Number of input features per particle: {input_dim}", flush=True, file=logfile
    )
    print("---------------", flush=True, file=logfile)

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
        log=args.full_kinematics or args.six_features,
    )
    if "from_scratch" not in args.label:
        # Load the pretrained model
        print("\nLoading the network", flush=True, file=logfile)
        if args.ep == -1:
            load_path = f"/ssl-jet-vol-v3/JetCLR/models/{args.label}/final_model.pt"
        elif args.ep == 0:
            load_path = f"/ssl-jet-vol-v3/JetCLR/models/{args.label}/model_best.pt"
        else:
            load_path = (
                f"/ssl-jet-vol-v3/JetCLR/models/{args.label}/model_ep{args.ep}.pt"
            )
        if "4gpu" in args.label:
            # Load the original state_dict
            state_dict = torch.load(load_path)

            # Create a new state_dict without the 'module.' prefix
            new_state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }

            # Load the new state_dict into your model
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(torch.load(load_path))
        print(f"Loaded model from {load_path}", flush=True, file=logfile)
    # initialize the MLP projector
    finetune_mlp_dim = args.output_dim
    if args.finetune_mlp:
        finetune_mlp_dim = f"{args.output_dim}-{args.finetune_mlp}"

    proj = Projector(2, finetune_mlp_dim).to(args.device)
    print(f"finetune mlp: {proj}", flush=True, file=logfile)
    if args.finetune:
        optimizer = optim.Adam(
            [{"params": proj.parameters()}, {"params": net.parameters(), "lr": 1e-6}],
            lr=1e-4,
        )
    else:
        net.eval()
        optimizer = optim.Adam(proj.parameters(), lr=1e-4)

    # send network to device
    net.to(device)

    loss = nn.CrossEntropyLoss(reduction="mean")

    l_val_best = 99999
    acc_val_best = 0
    rej_val_best = 0

    softmax = torch.nn.Softmax(dim=1)
    loss_train_all = []
    loss_val_all = []
    acc_val_all = []

    for epoch in range(args.n_epochs):
        # re-batch the data on each epoch
        indices_list = torch.split(torch.randperm(tr_dat.shape[0]), args.batch_size)
        indices_list_val = torch.split(torch.randperm(vl_dat.shape[0]), args.batch_size)

        # initialise timing stats
        te_start = time.time()
        te0 = time.time()

        # initialise lists to store batch stats
        losses_e = []
        losses_e_val = []
        predicted_e = []  # store the predicted labels by batch
        correct_e = []  # store the true labels by batch

        # the inner loop goes through the dataset batch by batch
        for i, indices in enumerate(indices_list):
            net.optimizer.zero_grad()
            x = tr_dat[indices, :, :]
            x = torch.Tensor(x).transpose(1, 2).to(args.device)
            y = tr_lab[indices]
            y = torch.Tensor(y).to(args.device)
            if args.finetune:
                net.train()
            proj.train()
            reps = net(x, use_mask=args.mask, use_continuous_mask=args.cmask)
            out = proj(reps)
            batch_loss = loss(out, y.long()).to(device)
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.detach().cpu().item()
            losses_e.append(batch_loss)
        loss_e = np.mean(np.array(losses_e))
        loss_train_all.append(loss_e)

        te1 = time.time()
        print(f"Training done in {te1-te0} seconds", flush=True, file=logfile)
        te0 = time.time()

        # validation
        with torch.no_grad():
            for i, indices in enumerate(indices_list_val):
                x = vl_dat[indices, :, :]
                x = torch.Tensor(x).transpose(1, 2).to(args.device)
                y = vl_lab[indices]
                y = torch.Tensor(y).to(args.device)
                if args.finetune:
                    net.train()
                proj.train()
                reps = net(x, use_mask=args.mask, use_continuous_mask=args.cmask)
                out = proj(reps)
                batch_loss = loss(out, y.long()).detach().cpu().item()
                losses_e_val.append(batch_loss)

                predicted_e.append(softmax(out).cpu().data.numpy())
                correct_e.append(y.cpu().data)
            loss_e_val = np.mean(np.array(losses_e_val))
            loss_val_all.append(loss_e_val)

        te1 = time.time()
        print(
            f"validation done in {round(te1-te0, 1)} seconds", flush=True, file=logfile
        )

        print(
            "epoch: "
            + str(epoch)
            + ", loss: "
            + str(round(loss_train_all[-1], 5))
            + ", val loss: "
            + str(round(loss_val_all[-1], 5)),
            flush=True,
            file=logfile,
        )

        # get the predicted labels and true labels
        predicted = np.concatenate(predicted_e)
        target = np.concatenate(correct_e)

        # get the accuracy
        accuracy = accuracy_score(target, predicted[:, 1] > 0.5)
        print(
            "epoch: " + str(epoch) + ", accuracy: " + str(round(accuracy, 5)),
            flush=True,
            file=logfile,
        )
        acc_val_all.append(accuracy)

        # save the latest model
        if args.finetune:
            torch.save(net.state_dict(), expt_dir + "simclr_finetune_last" + ".pt")
        torch.save(proj.state_dict(), expt_dir + "projector_finetune_last" + ".pt")

        # save the model if lowest val loss is achieved
        if loss_val_all[-1] < l_val_best:
            # print("new lowest val loss", flush=True, file=logfile)
            l_val_best = loss_val_all[-1]
            if args.finetune:
                torch.save(
                    net.state_dict(), expt_dir + "simclr_finetune_best_loss" + ".pt"
                )
            torch.save(
                proj.state_dict(), expt_dir + "projector_finetune_best_loss" + ".pt"
            )
            np.save(
                f"{expt_dir}validation_target_vals_loss.npy",
                target,
            )
            np.save(
                f"{expt_dir}validation_predicted_vals_loss.npy",
                predicted,
            )
        # also save the model if highest val accuracy is achieved
        if acc_val_all[-1] > acc_val_best:
            print("new highest val accuracy", flush=True, file=logfile)
            acc_val_best = acc_val_all[-1]
            if args.finetune:
                torch.save(
                    net.state_dict(), expt_dir + "simclr_finetune_best_acc" + ".pt"
                )
            torch.save(
                proj.state_dict(), expt_dir + "projector_finetune_best_acc" + ".pt"
            )
            np.save(
                f"{expt_dir}validation_target_vals_acc.npy",
                target,
            )
            np.save(
                f"{expt_dir}validation_predicted_vals_acc.npy",
                predicted,
            )
        # calculate the AUC and imtafe and output to the logfile
        auc, imtafe = get_perf_stats(target, predicted[:, 1])

        if imtafe > rej_val_best:
            print("new highest val rejection", flush=True, file=logfile)
            print(
                f"epoch: {epoch}, AUC: {auc}, IMTAFE: {imtafe}",
                flush=True,
                file=logfile,
            )
            rej_val_best = imtafe
            if args.finetune:
                torch.save(
                    net.state_dict(), expt_dir + "simclr_finetune_best_rej" + ".pt"
                )
            torch.save(
                proj.state_dict(), expt_dir + "projector_finetune_best_rej" + ".pt"
            )
            np.save(
                f"{expt_dir}validation_target_vals_rej.npy",
                target,
            )
            np.save(
                f"{expt_dir}validation_predicted_vals_rej.npy",
                predicted,
            )

        # save all losses and accuracies
        np.save(
            f"{expt_dir}loss_train.npy",
            np.array(loss_train_all),
        )
        np.save(
            f"{expt_dir}loss_val.npy",
            np.array(loss_val_all),
        )
        np.save(
            f"{expt_dir}acc_val.npy",
            np.array(acc_val_all),
        )
        te_end = time.time()
        print(
            f"epoch {epoch} done in {round(te_end - te_start, 1)} seconds",
            flush=True,
            file=logfile,
        )

    # Training done
    print("Training done", flush=True, file=logfile)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune-mlp",
        default="",
        help="Size and number of layers of the MLP finetuning head following output_dim of model, e.g. 512-256-128",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        action="store",
        dest="num_samples",
        default=100000,
        help="number of jets for training",
    )
    parser.add_argument(
        "--finetune",
        type=int,
        action="store",
        help="keep the transformer frozen and only train the MLP head",
    )
    parser.add_argument(
        "--trial",
        type=int,
        action="store",
        help="Trial number for the experiment",
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
        "--dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v3/toptagging/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        action="store",
        dest="num_files",
        default=1,
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
        default=1,
        help="use mask in transformer",
    )
    parser.add_argument(
        "--cmask",
        type=int,
        action="store",
        default=0,
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
        help="label of the model to load",
    )
    parser.add_argument(
        "--group-tag",
        type=str,
        action="store",
        dest="group_tag",
        default="",
        help="tag of the experiment group",
    )
    parser.add_argument(
        "--ep",
        type=int,
        action="store",
        dest="ep",
        default=-1,
        help="to use the model after which epoch",
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
        default=0,
        help="use the full 7 kinematic features instead of just 3",
    )
    parser.add_argument(
        "--six-features",
        type=int,
        action="store",
        dest="six_features",
        default=1,
        help="use the 6 kinematic features without delta R",
    )
    parser.add_argument(
        "--raw-3",
        type=int,
        action="store",
        dest="raw_3",
        default=0,
        help="use the 3 raw features",
    )

    args = parser.parse_args()
    main(args)
