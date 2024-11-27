#!/bin/env python3.7

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import random
import time
import glob
import argparse
import gc
sys.path.append('../')

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# seaborn
import seaborn as sns
from scipy.stats import pearsonr

# load custom modules required for jetCLR training
from src.modules.jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets
from src.modules.transformer import Transformer
from src.modules.losses import contrastive_loss, align_loss, uniform_loss
from src.modules.perf_eval import get_perf_stats, linear_classifier_test 
from src.modules.dataset import JetClassDataset

# set the number of threads that pytorch will use
torch.set_num_threads(2)

def create_balanced_dataset(test_dat_in, test_lab_in, signal_class):
    """
    Creates a dataset with a 1:1 signal to background ratio, evenly sampling from the background classes.
    
    Parameters:
    - test_dat_in: Input data, a PyTorch tensor.
    - test_lab_in: Corresponding input labels, a PyTorch tensor of shape [N, 10] where N is the total number of samples.
    - signal_class: The index of the signal class.
    
    Returns:
    - balanced_dat: Balanced dataset data.
    - balanced_lab: Balanced dataset labels.
    """
    # Identify signal samples
    signal_indices = (test_lab_in[:, signal_class] == 1).nonzero(as_tuple=True)[0]
    signal_dat = test_dat_in[signal_indices]
    signal_lab = test_lab_in[signal_indices]
    
    # Determine the number of samples to take from each background class
    num_signal_samples = signal_indices.size(0)
    num_background_classes = test_lab_in.size(1) - 1
    samples_per_background_class = num_signal_samples // num_background_classes
    
    # Collect background samples
    background_dat = []
    background_lab = []
    for class_idx in range(test_lab_in.size(1)):
        if class_idx == signal_class:
            continue
        class_indices = (test_lab_in[:, class_idx] == 1).nonzero(as_tuple=True)[0]
        selected_indices = class_indices[:samples_per_background_class]
        background_dat.append(test_dat_in[selected_indices])
        background_lab.append(test_lab_in[selected_indices])
    
    # Concatenate signal and background samples
    balanced_dat = torch.cat([signal_dat] + background_dat, dim=0)
    balanced_lab = torch.cat([signal_lab] + background_lab, dim=0)
    
    return balanced_dat, balanced_lab

def convert_label(lab_in, desired_label):
    labels = [
         'QCD',
         'Hbb',
         'Hcc',
         'Hgg',
         'H4q',
         'Hqql',
         'Zqq',
         'Wqq',
         'Tbqq',
         'Tbl'
            ]
    indices_to_modify = [i for i, label in enumerate(labels) if label != desired_label]
    lab_in_bin = lab_in.clone()
    lab_in_bin[:, indices_to_modify] = 0
    lab_in_max_indices = torch.argmax(lab_in_bin, dim=1)
    lab_in_collapsed = lab_in_bin[torch.arange(lab_in_bin.shape[0]), lab_in_max_indices]
    return lab_in_collapsed

def shuffle_data(tr_dat_in, tr_lab_in):
    # creating the training dataset
    print( "Splitting dataset into signal and background", flush=True )
    tr_bkg_dat = tr_dat_in[ tr_lab_in==0 ].clone()
    tr_sig_dat = tr_dat_in[ tr_lab_in==1 ].clone()
    return tr_bkg_dat, tr_sig_dat

def initialize_and_load_network(input_dim, args):
    """
    Initializes a Transformer network with given parameters contained within `args`, 
    except for `input_dim` which is provided separately. Sends the network to the appropriate 
    device (CUDA if available, otherwise CPU), and loads pre-trained state from a specified 
    file path contained in `args.load_path`.

    Parameters:
    - input_dim: Dimension of the input data.
    - args: An object (e.g., argparse.Namespace) containing the following attributes:
        - model_dim: Dimensionality of the model.
        - output_dim: Dimensionality of the output.
        - n_heads: Number of attention heads in the Transformer.
        - dim_feedforward: Dimensionality of the feedforward network in the Transformer.
        - n_layers: Number of layers in the Transformer.
        - learning_rate: Learning rate for the optimizer.
        - n_head_layers: Number of head layers in the Transformer.
        - opt: Optimizer type.
        - load_path: Path from which to load the pre-trained state dictionary.
        - full_kinematics: Boolean or parameter indicating whether to use full kinematics.
        - dropout: Dropout rate for the Transformer (optional, default=0.1).
    
    Returns:
    - net: The initialized and loaded Transformer network.
    - device: The device (CUDA or CPU) to which the network has been sent.
    """
    logfile = args.logfile
    print("initialising the network", file=logfile, flush=True)
    net = Transformer(input_dim, args.model_dim, args.output_dim, args.n_heads, 
                      args.dim_feedforward, args.n_layers, args.learning_rate, 
                      args.n_head_layers, dropout=0.1, opt=args.opt, 
                      log=args.full_kinematics)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.load_state_dict(torch.load(f"{args.load_path}"))
    print("All keys matched successfully", file=logfile, flush=True)
    return net

def obtain_representations(net, vl_dat_1, args):
    """
    Obtains representations from two validation datasets using the provided network,
    configured according to the given `args`. This function assumes the network has a 
    method `forward_batchwise` for processing data in batches. It marks the beginning 
    of the Linear Classifier Test (LCT) run after obtaining the representations.

    Parameters:
    - net: The neural network model to evaluate, assumed to have a `forward_batchwise` method.
    - vl_dat_1: The first set of validation data.
    - args: An object containing the following attributes:
        - batch_size: The size of the batches to use when processing the data.
        - mask: A flag indicating whether to use a mask during evaluation.
        - cmask: A flag indicating whether to use a continuous mask during evaluation.

    The function prints status updates, including when representations are being obtained
    and when the LCT is starting.
    """
    logfile = args.logfile
    # print("obtaining representations", file=logfile, flush=True)
    gc.collect()  # Optional garbage collection before starting

    # Evaluate the network on the testing data
    with torch.no_grad():
        net.eval()  # Set the network to evaluation mode
        vl_reps_1 = net.forward_batchwise(vl_dat_1.transpose(1,2), args.batch_size, 
                                          use_mask=args.mask, use_continuous_mask=args.cmask).detach().cpu().numpy()
        net.train()  # Set the network back to training mode
    
    # print("finished obtaining representations", file=logfile, flush=True)
    return vl_reps_1

def run_lct_and_plot_best_roc(vl_reps_1, vl_lab_1, vl_reps_2, vl_lab_2, linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, args, flag="full_features"):
    """
    Runs the Linear Classifier Test (LCT) multiple times for each representation layer, records AUC and IMTAFE metrics,
    selects the run with the highest AUC, and plots the ROC curve for that run.

    Parameters:
    - vl_reps_1: Validation representations set 1. Used to train the linear classifier
    - vl_lab_1: Validation labels set 1.
    - vl_reps_2: Validation representations set 2. Used to evaluate the linear classifier
    - vl_lab_2: Validation labels set 2.
    - linear_input_size: Input size for the linear classifier.
    - linear_batch_size: Batch size for training the linear classifier.
    - linear_n_epochs: Number of epochs for training the linear classifier.
    - linear_learning_rate: Learning rate for the linear classifier.
    - flag: to indicate whether this lct is done using the full features or the 8 pca features

    The function assumes the presence of `linear_classifier_test` to train the classifier,
    and `get_perf_stats` to obtain AUC and IMTAFE metrics.
    """
    logfile = args.logfile
    best_auc = 0
    best_imtafe = 0
    best_layer = None
    best_fpr = None
    best_tpr = None

    # Run LCT for each representation layer
    for run in range(5):
        for i in range(vl_reps_1.shape[1]):
            out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test(
                linear_input_size, linear_batch_size, linear_n_epochs, "adam", linear_learning_rate, 
                vl_reps_1[:,i,:], np.expand_dims(vl_lab_1, axis=1), vl_reps_2[:,i,:], np.expand_dims(vl_lab_2, axis=1)
            )
            auc, imtafe = get_perf_stats(out_lbs_f, out_dat_f)

            # Update best AUC if current AUC is higher
            if auc > best_auc:
                best_auc = auc
                best_layer = i
                fpr, tpr, thresholds = roc_curve(out_lbs_f, out_dat_f)
                best_fpr, best_tpr = fpr, tpr
            if imtafe > best_imtafe:
                best_imtafe = imtafe

            # Optional: Print performance stats for each layer and run
            print(f"Run {run+1}, Rep Layer {i}, AUC: {round(auc, 4)}, IMTAFE: {round(imtafe, 1)}", file=logfile, flush=True)

    # Plot the ROC curve for the run with the highest AUC
    print(f"Best AUC: {round(best_auc, 4)}, Best IMTAFE: {round(best_imtafe, 4)} from Rep Layer {best_layer}", file=logfile, flush=True)
    plt.figure(figsize=(8, 6))
    plt.plot(best_fpr, best_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % best_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best ROC Curve across All Runs and Layers')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    save_path = f"{args.save_dir}/roc_{flag}.png"
    plt.savefig(save_path)
    print(f"Roc curve for {flag} saved to: {save_path}", file=logfile, flush=True)

def do_pca(X):
    # Assuming X is your 1000-dimensional dataset with shape (n_samples, 1000)
    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    # Step 2: Apply PCA to reduce dimensionality to 8 components
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_standardized)
    # Now X_pca has the transformed data in the reduced dimensionality space
    return X_pca

def do_pca_multirep(X):
    """
    Applies PCA to reduce the dimensionality of a dataset with multiple sets of features (representations)
    for each sample. The input is a tensor/array of shape (num_samples, num_reps, num_features), and the output
    is of shape (num_samples, num_reps, 8), where each set of features is reduced to 8 principal components.

    Parameters:
    - X: A numpy array or tensor with shape (num_samples, num_reps, num_features) representing the dataset
         with multiple representations for each sample.

    Returns:
    - X_pca: A numpy array with shape (num_samples, num_reps, 8) where the dimensionality of each set of
             features has been reduced to 8 principal components.
    """
    num_samples, num_reps, num_features = X.shape
    X_pca = np.zeros((num_samples, num_reps, 8))  # Prepare the output array
    
    for rep in range(num_reps):
        # Extract the current set of representations
        X_rep = X[:, rep, :]
        
        # Standardize the data
        scaler = StandardScaler()
        X_rep_standardized = scaler.fit_transform(X_rep)
        
        # Apply PCA to reduce dimensionality to 8 components
        pca = PCA(n_components=8)
        X_rep_pca = pca.fit_transform(X_rep_standardized)
        
        # Store the reduced data in the output array
        X_pca[:, rep, :] = X_rep_pca
    
    return X_pca


def load_and_plot_metrics(args):
    """
    Attempts to load various metric arrays from specified paths based on a label,
    plots each metric if available, adding X and Y axis labels, and saves each plot to a designated directory.

    Parameters:
    - args
        - label: The label that identifies the experiment.
        - save_dir: The directory where the plots will be saved.
    """
    label = args.label
    save_dir = args.save_dir
    logfile = args.logfile
    base_path = f"/ssl-jet-vol-v2/JetCLR/models/{label}/"
    metrics = {
        "auc_epochs": ("LCT AUC vs epochs", "Epochs", "AUC"),
        "imtafe_epochs": ("LCT imtafe vs epochs", "Epochs", "IMTAFE"),
        "align_loss_train": ("Alignment loss vs epochs", "Epochs", "Alignment Loss"),
        "clr_losses": ("Total CLR losses vs epochs", "Epochs", "CLR Loss"),
        "uniform_loss_train": ("Uniform loss vs epochs", "Epochs", "Uniform Loss"),
        "val_losses": ("Validation losses vs epochs", "Epochs", "Validation Loss")
    }
    
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for metric_name, (plot_title, xlabel, ylabel) in metrics.items():
        try:
            metric_values = np.load(base_path + f"{metric_name}.npy")
            plt.figure()
            plt.plot(metric_values)
            plt.title(plot_title)
            plt.xlabel(xlabel)  # Set X axis label
            plt.ylabel(ylabel)  # Set Y axis label
            
            save_path = os.path.join(save_dir, f"{metric_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Plot saved to: {save_path}", file=logfile, flush=True)
        except IOError as e:
            print(f"Error loading {metric_name}.npy: {e}", file=logfile, flush=True)
            
def mean_pearson_coefficient(data):
    """
    Calculates the mean Pearson correlation coefficient across all unique pairs of features in the dataset.

    Parameters:
    - data: A 2D numpy array where rows represent samples and columns represent features.

    Returns:
    - mean_corr: The mean Pearson correlation coefficient across all unique feature pairs.
    """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    # Extract the upper triangular part of the correlation matrix, excluding the diagonal
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    
    # Compute the mean of the upper triangular values
    mean_corr = np.mean(upper_tri)
    
    return mean_corr

def plot_pair_plots(data, args, flag):
    """
    Generates pair plots for the QCD dataset, including histograms on the diagonal 
    and scatter plots with Pearson correlation coefficients for off-diagonal elements.
    
    Parameters:
    - data: A numpy array where rows represent samples and columns represent features.
    - args
    - flag: QCD or Top
    
    Saves the generated plot to the specified path constructed from the provided parameters.
    """
    logfile = args.logfile
    num_feats = data.shape[1]
    fig, axs = plt.subplots(num_feats, num_feats, figsize=(30, 30))  # Increase figure size

    for i in range(num_feats):
        for j in range(i + 1):
            # Compute Pearson correlation coefficient using scipy
            corr_coeff, _ = pearsonr(data[:, i], data[:, j])
            
            # Diagonal histograms
            if i == j:
                axs[i, j].hist(data[:, i], bins=20, color='skyblue', edgecolor='black')
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
            else:
                # Scatter plot for off-diagonal
                axs[i, j].scatter(data[:, j], data[:, i], s=5)
                axs[i, j].set_title(f"{corr_coeff:.2f}", fontsize=8, pad=-15)  # Decrease font size
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[j, i].axis('off')  # Hide the upper part of the matrix

    fig.suptitle(f'{flag} Pair Plots', fontsize=20)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust spacing between subplots
    save_path = f"{args.save_dir}/{flag}_pair_plots.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Pair plots saved to: {save_path}", file=logfile, flush=True)

def plot_top_and_qcd_features(data_top, data_qcd, args):
    """
    Generates histograms for each feature from two datasets, 'top' and 'QCD',
    on the same canvas for direct comparison.

    Parameters:
    - data_top: A numpy array for the 'top' dataset where rows represent samples and columns represent features.
    - data_qcd: A numpy array for the 'QCD' dataset where rows represent samples and columns represent features.
    - args
    """
    logfile = args.logfile
    num_feats = data_qcd.shape[1]
    num_columns = 2
    num_rows = int(np.ceil(num_feats / num_columns))

    # Adjust the figure size
    subplot_width = 6
    subplot_height = 4
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(subplot_width * num_columns, subplot_height * num_rows))

    # Style settings
    sns.set_style("whitegrid")
    colors = ['blue', 'red']

    for i in range(num_feats):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col] if num_rows > 1 else axs[col]

        sns.histplot(data_top[:, i], bins=50, ax=ax, color=colors[0], label='Top', alpha=0.6)
        sns.histplot(data_qcd[:, i], bins=50, ax=ax, color=colors[1], label='QCD', alpha=0.6)

        ax.set_title(f"PCA Feature {i}")
        ax.legend()

        sns.despine(ax=ax)

    # If the number of features isn't a multiple of the columns, remove unused subplots
    if num_feats % num_columns != 0:
        for j in range(num_feats, num_rows * num_columns):
            axs.flatten()[j].axis('off')

    fig.suptitle('Top and QCD Feature Comparison', y=1.02)
    plt.tight_layout(pad=2.0)
    save_path = f"{args.save_dir}/Top_and_QCD_Feature_Comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plots saved to: {save_path}", file=logfile, flush=True)

def plot_tsne_visualization(data_numpy, labels_numpy, args, perplexity=30, n_iter=300, s=1):
    """
    Applies t-SNE to the provided dataset and plots the 2D visualization,
    differentiating points by labels ('top' and 'QCD').

    Parameters:
    - data_numpy: A numpy array where rows represent samples and columns represent features.
    - labels_numpy: A numpy array of labels corresponding to the samples in data_numpy.
    - save_dir: Directory where the plot image will be saved.
    - perplexity: The perplexity hyperparameter for t-SNE (default 30).
    - n_iter: The number of iterations for optimization in t-SNE (default 300).
    - s: Size of the points in the scatter plot (default 1).
    """
    save_dir = args.save_dir
    logfile = args.logfile
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data_numpy)

    # Separate points by labels
    top_points = tsne_results[labels_numpy == 1]
    qcd_points = tsne_results[labels_numpy == 0]

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(top_points[:, 0], top_points[:, 1], color='b', alpha=0.5, label='Top', s=s)
    plt.scatter(qcd_points[:, 0], qcd_points[:, 1], color='r', alpha=0.5, label='QCD', s=s)
    plt.title("t-SNE Visualization of Jet Features")
    plt.legend(loc='upper right')
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.grid(False)
    
    # Save and show plot
    save_path = f"{save_dir}/t-SNE.png"
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plots saved to: {save_path}", file=logfile, flush=True)

def main(args):
    # -------------------------------
    #               Set up
    # -------------------------------
    t0 = time.time()
    args.save_dir = f"/ssl-jet-vol-v2/JetCLR/models/model_performances/JetClass/{args.label}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # initialise logfile
    logfile_path = f"{args.save_dir}/eval-log.txt"
    logfile = open(logfile_path, "a")
    args.logfile = logfile
    print("logfile initialised", file=logfile, flush=True)
    
    print("Begin set up", file=logfile, flush=True)
    args.nconstit = 50
    args.n_heads = 4
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128
    args.load_path = f"/ssl-jet-vol-v2/JetCLR/models/{args.label}/model_ep{args.epoch}.pt"

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

    # -------------------------------
    #        Load data and model
    # -------------------------------
    print( "loading data", file=logfile, flush=True)
    dataset_path = "/ssl-jet-vol-v2/JetClass/processed/raw"
    test_dataset = JetClassDataset(
            dataset_path,
            flag="test",
            args=args,
            logfile=logfile,
            load_labels=True,
        )
    # load the test data
    test_dat_in, test_lab_in = test_dataset.fetch_entire_dataset()

    balanced_dat, balanced_lab = create_balanced_dataset(test_dat_in, test_lab_in, signal_class=8)
    # Shuffle the input data and labels together
    indices = torch.randperm(balanced_dat.size(0))
    shuf_dat = balanced_dat[indices]
    shuf_lab = balanced_lab[indices]
    
    shuf_lab = convert_label(shuf_lab, 'Tbqq')

    input_dim = test_dataset.get_sample_shape()[0]
    print(f"input_dim: {input_dim}", file=logfile, flush=True)

    t1 = time.time()

    print(
        "time taken to load and preprocess data: "
        + str(np.round(t1 - t0, 2))
        + " seconds",
        flush=True,
        file=logfile,
    )

    # Load pretrained model
    net = initialize_and_load_network(input_dim, args)

    print("-------------------------------", file=logfile, flush=True)
    print("      Plot losses and metrics", file=logfile, flush=True)
    print("-------------------------------\n", file=logfile, flush=True)
    t_start = time.time()
    
    
    load_and_plot_metrics(args)

    t_end = time.time()
    print(f"time taken to Plot losses and metrics: {np.round(t_end - t_start, 2)} seconds", flush=True, file=logfile)

    print("-------------------------------", file=logfile, flush=True)
    print("      Pair Plots", file=logfile, flush=True)
    print("-------------------------------\n", file=logfile, flush=True)
    t_start = time.time()
    
    data_qcd, data_top = shuffle_data(shuf_dat, shuf_lab)
    reps_top_full = obtain_representations(net, data_top, args)
    reps_qcd_full = obtain_representations(net, data_qcd, args)
    # do PCA
    reps_top_full_pca = do_pca_multirep(reps_top_full)
    reps_qcd_full_pca = do_pca_multirep(reps_qcd_full)
    # just want to use the 0th rep (i.e. directly from the transformer) for now
    reps_top = reps_top_full[:, 1, :]
    reps_qcd = reps_qcd_full[:, 1, :]
    reps_top_pca = reps_top_full_pca[:, 1, :]
    reps_qcd_pca = reps_qcd_full_pca[:, 1, :]

    # calculate the mean Pearson Correlation Coefficient for both
    print(f"Mean Pearson Coefficient for Top: {round(mean_pearson_coefficient(reps_top), 3)}", flush=True, file=logfile)
    print(f"Mean Pearson Coefficient for QCD: {round(mean_pearson_coefficient(reps_qcd), 3)}", flush=True, file=logfile)
    
    # Produce pair plots
    plot_pair_plots(reps_qcd_pca, args, "QCD")
    plot_pair_plots(reps_top_pca, args, "Top")

    # Plot overlaid top and qcd feature distribution
    plot_top_and_qcd_features(reps_top_pca, reps_qcd_pca, args)

    t_end = time.time()
    print(f"time taken to produce Pair Plots: {np.round(t_end - t_start, 2)} seconds", flush=True, file=logfile)

    print("-------------------------------", file=logfile, flush=True)
    print("      Producing t-SNE plots", file=logfile, flush=True)
    print("-------------------------------\n", file=logfile, flush=True)
    t_start = time.time()
    
    reps_full = obtain_representations(net, shuf_dat, args)[:, 1, :]
    plot_tsne_visualization(reps_full, shuf_lab, args)

    t_end = time.time()
    print(f"time taken to produce t-SNE: {np.round(t_end - t_start, 2)} seconds", flush=True, file=logfile)
    
    print("-------------------------------", file=logfile, flush=True)
    print("      LCT", file=logfile, flush=True)
    print("-------------------------------\n", file=logfile, flush=True)
    t_start = time.time()
    # Create two validation sets for LCT
    test_dat_1 = shuf_dat[: int(len(shuf_dat) / 2)]
    test_lab_1 = shuf_lab[: int(len(shuf_lab) / 2)]
    test_dat_2 = shuf_dat[int(len(shuf_dat) / 2) :]
    test_lab_2 = shuf_lab[int(len(shuf_lab) / 2) :]
    
    # set-up parameters for the LCT
    linear_input_size = args.output_dim
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 128

    # LCT
    print("\nStarting LCT", file=logfile, flush=True)
    test_reps_1 = obtain_representations(net, test_dat_1, args)
    test_reps_2 = obtain_representations(net, test_dat_2, args)
    run_lct_and_plot_best_roc(test_reps_1, test_lab_1.detach().numpy(), test_reps_2, test_lab_1.detach().numpy(), linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, args, flag="full_features")
    # run LCT with pca features
    run_lct_and_plot_best_roc(test_reps_1_pca, test_lab_1.detach().numpy(), test_reps_2_pca, test_lab_1.detach().numpy(), 8, linear_batch_size, linear_n_epochs, linear_learning_rate, args, flag = "pca")
    t_end = time.time()
    print(f"time taken to do LCT: {np.round(t_end - t_start, 2)} seconds", flush=True, file=logfile)
    print("All evaluation done", flush=True, file=logfile)

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
    parser.add_argument(
        "--percent",
        type=int,
        action="store",
        dest="percent",
        default=1,
        help="percentage of data to use for training",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        action="store",
        default=50,
        help="use the model saved after which epoch",
    )


    args = parser.parse_args()
    main(args)
