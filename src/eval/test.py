# load standard python modules
import sys

sys.path.insert(0, "../")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import argparse
import copy
from tqdm import tqdm
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn import metrics

plt.style.use(hep.style.ROOT)

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# load custom modules required for jetCLR training
from src.modules.transformer import Transformer
from src.modules.losses import contrastive_loss, align_loss, uniform_loss
from src.modules.perf_eval import get_perf_stats, linear_classifier_test


# load data
def load_data(dataset_path, flag, n_files=-1):
    # make another variable that combines flag and subdirectory such as 3_features_raw
    path_id = f"{flag}-"
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/6_features_raw/data/*")
    path_id += "6_features"

    data = []
    for i, _ in enumerate(data_files):
        data.append(
            np.load(f"{dataset_path}/{flag}/processed/6_features_raw/data/data_{i}.npy")
        )
        print(f"--- loaded file {i} from `{path_id}` directory")
        if n_files != -1 and i == n_files - 1:
            break
    return data


def load_labels(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/6_features_raw/labels/*")

    data = []
    for i, file in enumerate(data_files):
        data.append(
            np.load(
                f"{dataset_path}/{flag}/processed/6_features_raw/labels/labels_{i}.npy"
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


def get_perf_stats(labels, measures, sig=0.5):
    measures = np.nan_to_num(measures)
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, thresholds = metrics.roc_curve(labels, measures)
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= sig]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= sig]
    try:
        imtafe = np.nan_to_num(
            1 / fpr2[list(tpr2).index(find_nearest(list(tpr2), sig))]
        )
    except:
        imtafe = 1
    return auc, imtafe


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


def main(args):
    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}", flush=True)
    else:
        device = torch.device("cpu")
        print("Device: CPU", file=logfile, flush=True)

    t0 = time.time()
    args.save_dir = (
        f"/ssl-jet-vol-v2/JetCLR/models/model_performances/finetuning/{args.label}"
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # initialise logfile
    logfile_path = f"{args.save_dir}/eval-log.txt"
    logfile = open(logfile_path, "a")
    args.logfile = logfile
    print("logfile initialised", file=logfile, flush=True)

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
    # initialize and load the MLP projector
    finetune_mlp_dim = args.output_dim
    if args.finetune_mlp:
        finetune_mlp_dim = f"{args.output_dim}-{args.finetune_mlp}"

    proj = Projector(2, finetune_mlp_dim).to(args.device)
    print(f"finetune mlp: {proj}", flush=True)

    def load_model(metric="best_acc"):
        # Load the pretrained model
        print("\nLoading the network", flush=True)
        model_load_path = f"/ssl-jet-vol-v3/JetCLR/models/finetuning/{args.label}/simclr_finetune_{metric}.pt"
        net.load_state_dict(torch.load(model_load_path))

        projector_load_path = f"/ssl-jet-vol-v3/JetCLR/models/finetuning/{args.label}/projector_finetune_{metric}.pt"
        proj.load_state_dict(torch.load(projector_load_path))
        # send network to device
        net.to(device)

    load_model("best_loss")

    # load testing data
    data = load_data("/ssl-jet-vol-v3/toptagging", "test", 4)
    labels = load_labels("/ssl-jet-vol-v3/toptagging", "test", 4)
    # data = load_data("/ssl-jet-vol-v3/toptagging", "val", 1)
    # labels = load_labels("/ssl-jet-vol-v3/toptagging", "val", 1)
    tr_dat_in = np.concatenate(data, axis=0)  # Concatenate along the first axis
    tr_lab_in = np.concatenate(labels, axis=0)
    # tr_dat_in = tr_dat_in[0:10000]
    # tr_lab_in = tr_lab_in[0:10000]

    # creating the training dataset
    args.sbratio = 1.0
    print("shuffling data and doing the S/B split")
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

    input_dim = tr_dat.shape[1]
    print(f"input_dim: {input_dim}")

    softmax = torch.nn.Softmax(dim=1)
    indices_list = torch.split(torch.randperm(tr_dat.shape[0]), args.batch_size)
    predicted_e = []  # store the predicted labels by batch
    correct_e = []  # store the true labels by batch
    net.eval()
    proj.eval()
    for metric in ["best_acc", "best_rej", "best_loss", "last"]:
        load_model(metric)
        with torch.no_grad():
            for i, indices in enumerate(
                tqdm(indices_list, desc="Processing", unit="batch")
            ):
                x = tr_dat[indices, :, :]
                x = torch.Tensor(x).transpose(1, 2).to(args.device)
                y = tr_lab[indices]
                y = torch.Tensor(y).to(args.device)
                reps = net(x, use_mask=True, use_continuous_mask=False)
                out = proj(reps)

                predicted_e.append(softmax(out).cpu().data.numpy())
                correct_e.append(y.cpu().data)
            predicted = np.concatenate(predicted_e)
            target = np.concatenate(correct_e)

            # get the accuracy
            accuracy = accuracy_score(target, predicted[:, 1] > 0.5)
            auc, rej_50 = get_perf_stats(target, predicted[:, 1], sig=0.5)
            auc, rej_30 = get_perf_stats(target, predicted[:, 1], sig=0.3)
            print(
                f"Model with {metric}: Accuracy: {accuracy}, AUC: {auc}, rej_50: {rej_50}, rej_30: {rej_30}",
                file=logfile,
                flush=True,
            )
            # save the predited and true labels
            np.save(f"{args.save_dir}/predicted_{metric}.npy", predicted)
            np.save(f"{args.save_dir}/target_{metric}.npy", target)


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
        default=1,
        help="number of files for testing",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        action="store",
        dest="model_dim",
        default=1024,
        help="dimension of the transformer-encoder",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        action="store",
        dest="dim_feedforward",
        default=1024,
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
        default=1024,
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
        default=512,
        help="batch_size",
    )
    parser.add_argument(
        "--full-kinematics",
        type=int,
        action="store",
        dest="full_kinematics",
        default=1,
        help="use the full 7 kinematic features instead of just 3",
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
