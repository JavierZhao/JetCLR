#!/bin/env python3.7

# load standard python modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
import glob

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules required for jetCLR training
from .modules.jet_augs import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets
from .modules.transformer import Transformer
from .modules.losses import contrastive_loss, align_loss, uniform_loss
from .modules.perf_eval import get_perf_stats, linear_classifier_test 

# import args from extargs.py file
# import extargs as args
def print_and_log(message, file=None, flush=True):
    print(message)
    if file is not None:
        print(message, file=file, flush=flush)

# load the datafiles
def load_data(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/data/*")
#     print_and_log(data_files)

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/3_features/data/data_{i}.pt")
        print_and_log(f"--- loaded file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def load_labels(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/labels/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/3_features/labels/labels_{i}.pt")
        print_and_log(f"--- loaded label file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data

# set the number of threads that pytorch will use
torch.set_num_threads(2)

def main(args):
    t0 = time.time()
    args.logfile = f"/ssl-jet-vol-v2/JetCLR/logs/zz-simCLR-{args.label}-log.txt"
    args.tr_dat_path = f"/ssl-jet-vol-v2/toptagging/train/processed/3_features/"
    args.tr_lab_path = args.tr_dat_path
    args.nconstit = 50
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128

    # initialise logfile
    logfile = open( args.logfile, "a" )
    print_and_log( "logfile initialised", file=logfile, flush=True )

    # set gpu device
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print_and_log( "device: " + str( device ), flush=True, file=logfile )

    # set up results directory
    base_dir = "/ssl-jet-vol-v2/JetCLR/models/"
    expt_tag = args.label
    expt_dir = base_dir + "experiments/" + expt_tag + "/"

<<<<<<< HEAD
# check if experiment already exists and is not empty
if os.path.isdir(expt_dir) and os.listdir(expt_dir):
    sys.exit("ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake")
else:
    # This will create the directory if it does not exist or if it is empty
    os.makedirs(expt_dir, exist_ok=True)
print("experiment: "+str(args.expt), file=logfile, flush=True)
=======
    # check if experiment already exists
    if os.path.isdir(expt_dir):
        sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")
    else:
        os.makedirs(expt_dir)
    print_and_log("experiment: "+str(args.label), file=logfile, flush=True)
>>>>>>> 7873bccfdff0bf112347c38e26eb70a3ff0574a2

    # load data
    print_and_log( "loading data", flush=True, file=logfile )
    data = load_data("/ssl-jet-vol-v2/toptagging", "train", args.num_train_files)
    labels = load_labels("/ssl-jet-vol-v2/toptagging", "train", args.num_train_files)
    tr_dat_in = np.stack(data)
    tr_lab_in = np.stack(labels)

    # input dim to the transformer -> (pt,eta,phi)
    input_dim = tr_dat_in.shape[2]

    # creating the training dataset
    print_and_log( "shuffling data and doing the S/B split", flush=True, file=logfile )
    tr_bkg_dat = tr_dat_in[ tr_lab_in==0 ].copy()
    tr_sig_dat = tr_dat_in[ tr_lab_in==1 ].copy()
    nbkg_tr = int( tr_bkg_dat.shape[0] )
    nsig_tr = int( args.sbratio * nbkg_tr )
    list_tr_dat = list( tr_bkg_dat[ 0:nbkg_tr ] ) + list( tr_sig_dat[ 0:nsig_tr ] )
    list_tr_lab = [ 0 for i in range( nbkg_tr ) ] + [ 1 for i in range( nsig_tr ) ]
    ldz_tr = list( zip( list_tr_dat, list_tr_lab ) )
    random.shuffle( ldz_tr )
    tr_dat, tr_lab = zip( *ldz_tr )
    # reducing the training data
    # tr_dat = np.array( tr_dat )[0:100000]
    # tr_lab = np.array( tr_lab )[0:100000]

    # create two validation sets: 
    # one for training the linear classifier test (LCT)
    # and one for testing on it
    # we will do this just with tr_dat_in, but shuffled and split 50/50
    # this should be fine because the jetCLR training doesn't use labels
    # we want the LCT to use S/B=1 all the time
    list_vl_dat = list( tr_dat_in.copy() )
    list_vl_lab = list( tr_lab_in.copy() )
    ldz_vl = list( zip( list_vl_dat, list_vl_lab ) )
    random.shuffle( ldz_vl )
    vl_dat, vl_lab = zip( *ldz_vl )
    vl_dat = np.array( vl_dat )
    vl_lab = np.array( vl_lab )
    vl_len = vl_dat.shape[0]
    vl_split_len = int( vl_len/2 )
    vl_dat_1 = vl_dat[ 0:vl_split_len ]
    vl_lab_1 = vl_lab[ 0:vl_split_len ]
    vl_dat_2 = vl_dat[ -vl_split_len: ]
    vl_lab_2 = vl_lab[ -vl_split_len: ]

    # cropping all jets to a fixed number of consituents
    tr_dat = crop_jets( tr_dat, args.nconstit )
    vl_dat_1 = crop_jets( vl_dat_1, args.nconstit )
    vl_dat_2 = crop_jets( vl_dat_2, args.nconstit )

    # print_and_log data dimensions
    print_and_log( "training data shape: " + str( tr_dat.shape ), flush=True, file=logfile )
    print_and_log( "validation-1 data shape: " + str( vl_dat_1.shape ), flush=True, file=logfile )
    print_and_log( "validation-2 data shape: " + str( vl_dat_2.shape ), flush=True, file=logfile )
    print_and_log( "training labels shape: " + str( tr_lab.shape ), flush=True, file=logfile )
    print_and_log( "validation-1 labels shape: " + str( vl_lab_1.shape ), flush=True, file=logfile )
    print_and_log( "validation-2 labels shape: " + str( vl_lab_2.shape ), flush=True, file=logfile )

    t1 = time.time()

    # re-scale test data, for the training data this will be done on the fly due to the augmentations
    vl_dat_1 = rescale_pts( vl_dat_1 )
    vl_dat_2 = rescale_pts( vl_dat_2 )

    print_and_log( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds", flush=True, file=logfile )

    # set-up parameters for the LCT
    linear_input_size = args.output_dim
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 128

    print_and_log( "--- contrastive learning transformer network architecture ---", flush=True, file=logfile )
    print_and_log( "model dimension: " + str( args.model_dim ) , flush=True, file=logfile )
    print_and_log( "number of heads: " + str( args.n_heads ) , flush=True, file=logfile )
    print_and_log( "dimension of feedforward network: " + str( args.dim_feedforward ) , flush=True, file=logfile )
    print_and_log( "number of layers: " + str( args.n_layers ) , flush=True, file=logfile )
    print_and_log( "number of head layers: " + str( args.n_head_layers ) , flush=True, file=logfile )
    print_and_log( "optimiser: " + str( args.opt ) , flush=True, file=logfile )
    print_and_log( "mask: " + str( args.mask ) , flush=True, file=logfile )
    print_and_log( "continuous mask: " + str( args.cmask ) , flush=True, file=logfile )
    print_and_log( "--- hyper-parameters ---", flush=True, file=logfile )
    print_and_log( "learning rate: " + str( args.learning_rate ) , flush=True, file=logfile )
    print_and_log( "batch size: " + str( args.batch_size ) , flush=True, file=logfile )
    print_and_log( "temperature: " + str( args.temperature ) , flush=True, file=logfile )
    print_and_log( "--- symmetries/augmentations ---", flush=True, file=logfile )
    print_and_log( "rotations: " + str( args.rot ) , flush=True, file=logfile )
    print_and_log( "low pT smearing: " + str( args.ptd ) , flush=True, file=logfile )
    print_and_log( "pT smearing clip parameter: " + str( args.ptcm ) , flush=True, file=logfile )
    print_and_log( "translations: " + str( args.trs ) , flush=True, file=logfile )
    print_and_log( "translations width: " + str( args.trsw ) , flush=True, file=logfile )
    print_and_log( "---", flush=True, file=logfile )

    # initialise the network
    print_and_log( "initialising the network", flush=True, file=logfile )
    net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_layers, dropout=0.1, opt=args.opt )

    # send network to device
    net.to( device )

    # set learning rate scheduling, if required
    # SGD with cosine annealing
    if args.opt == "sgdca":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts( net.optimizer, 15, T_mult=2, eta_min=0, last_epoch=-1, verbose=False)
    # SGD with step-reduced learning rates
    if args.opt == "sgdslr":
        scheduler = torch.optim.lr_scheduler.StepLR( net.optimizer, 100, gamma=0.6, last_epoch=-1, verbose=False)

    # THE TRAINING LOOP

    print_and_log( "starting training loop, running for " + str( args.n_epochs ) + " epochs", flush=True, file=logfile )
    print_and_log( "---", flush=True, file=logfile )

    # initialise lists for storing training stats
    auc_epochs = []
    imtafe_epochs = []
    losses = []
    loss_align_epochs = []
    loss_uniform_epochs = []

    # cosine annealing requires per-batch calls to the scheduler, we need to know the number of batches per epoch
    if args.opt == "sgdca":
        # number of iterations per epoch
        iters = int( tr_dat.shape[0]/args.batch_size )
        print_and_log( "number of iterations per epoch: " + str(iters), flush=True, file=logfile )

    # the loop
    for epoch in range( args.n_epochs ):

        # re-batch the data on each epoch
        indices_list = torch.split( torch.randperm( tr_dat.shape[0] ), args.batch_size )

        # initialise timing stats
        te0 = time.time()
        
        # initialise lists to store batch stats
        loss_align_e = []
        loss_uniform_e = []
        losses_e = []

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
        for i, indices in enumerate( indices_list ):
            net.optimizer.zero_grad()
            x_i = tr_dat[indices,:,:]
            time1 = time.time()
            x_i = rotate_jets( x_i )
            x_j = x_i.copy()
            if args.rot:
                x_j = rotate_jets( x_j )
            time2 = time.time()
            if args.cf:
                x_j = collinear_fill_jets( x_j )
                x_j = collinear_fill_jets( x_j )
            time3 = time.time()
            if args.ptd:
                x_j = distort_jets( x_j, strength=args.ptst, pT_clip_min=args.ptcm )
            time4 = time.time()
            if args.trs:
                x_j = translate_jets( x_j, width=args.trsw )
                x_i = translate_jets( x_i, width=args.trsw )
            time5 = time.time()
            x_i = rescale_pts( x_i )
            x_j = rescale_pts( x_j )
            x_i = torch.Tensor( x_i ).transpose(1,2).to( device )
            x_j = torch.Tensor( x_j ).transpose(1,2).to( device )
            time6 = time.time()
            z_i = net( x_i, use_mask=args.mask, use_continuous_mask=args.cmask )
            z_j = net( x_j, use_mask=args.mask, use_continuous_mask=args.cmask )
            time7 = time.time()

            # calculate the alignment and uniformity loss for each batch
            if epoch%10==0:
                loss_align = align_loss( z_i, z_j )
                loss_uniform_zi = uniform_loss( z_i )
                loss_uniform_zj = uniform_loss( z_j )
                loss_align_e.append( loss_align.detach().cpu().numpy() )
                loss_uniform_e.append( ( loss_uniform_zi.detach().cpu().numpy() + loss_uniform_zj.detach().cpu().numpy() )/2 )
            time8 = time.time()

            # compute the loss, back-propagate, and update scheduler if required
            loss = contrastive_loss( z_i, z_j, args.temperature ).to( device )
            loss.backward()
            net.optimizer.step()
            if args.opt == "sgdca":
                scheduler.step( epoch + i / iters )
            losses_e.append( loss.detach().cpu().numpy() )
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

        loss_e = np.mean( np.array( losses_e ) )
        losses.append( loss_e )

        if args.opt == "sgdslr":
            scheduler.step()

        te1 = time.time()

        print_and_log( "epoch: " + str( epoch ) + ", loss: " + str( round(losses[-1], 5) ), flush=True, file=logfile )
        if args.opt == "sgdca" or args.opt == "sgdslr":
            print_and_log( "lr: " + str( scheduler._last_lr ), flush=True, file=logfile )
        print_and_log( f"total time taken: {round( te1-te0, 1 )}s, augmentation: {round(td1+td2+td3+td4+td5,1)}s, forward {round(td6, 1)}s, backward {round(td8, 1)}s, other {round(te1-te0-(td1+td2+td3+td4+td6+td7+td8), 2)}s", flush=True, file=logfile )

        # check memory stats on the gpu
        if epoch % 10 == 0:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside reserved
            print_and_log( f"CUDA memory: total {t}, reserved {r}, allocated {a}, free {f}", flush=True, file=logfile )

        # summarise alignment and uniformity stats
        if epoch%10==0:
            loss_align_epochs.append( np.mean( np.array( loss_align_e ) ) )
            loss_uniform_epochs.append( np.mean( np.array( loss_uniform_e ) ) )
            print_and_log( "alignment: " + str( loss_align_epochs[-1] ) + ", uniformity: " + str( loss_uniform_epochs[-1] ), flush=True, file=logfile )

        # check number of threads being used
        if epoch%10==0:
            print_and_log( "num threads in use: " + str( torch.get_num_threads() ), flush=True, file=logfile )

        # run a short LCT
        if epoch%10==0:
            print_and_log( "--- LCT ----" , flush=True, file=logfile )
            if args.trs:
                vl_dat_1 = translate_jets( vl_dat_1, width=args.trsw )
                vl_dat_2 = translate_jets( vl_dat_2, width=args.trsw )
            # get the validation reps
            with torch.no_grad():
                net.eval()
                #vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
                #vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
                vl_reps_1 = net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
                vl_reps_2 = net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
                net.train()
            # running the LCT on each rep layer
            auc_list = []
            imtafe_list = []
            # loop through every representation layer
            for i in range(vl_reps_1.shape[1]):   
                # just want to use the 0th rep (i.e. directly from the transformer) for now
                if i == 1:
                    vl0_test = time.time()
                    out_dat_vl, out_lbs_vl, losses_vl = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, "adam", linear_learning_rate, vl_reps_1[:,i,:], vl_lab_1, vl_reps_2[:,i,:], vl_lab_2 )
                    auc, imtafe = get_perf_stats( out_lbs_vl, out_dat_vl )
                    auc_list.append( auc )
                    imtafe_list.append( imtafe )
                    vl1_test = time.time()
                    print_and_log( "LCT layer " + str(i) + "- time taken: " + str( np.round( vl1_test - vl0_test, 2 ) ), flush=True, file=logfile )
                    print_and_log( "auc: " + str( np.round( auc, 4 ) ) + ", imtafe: " + str( round( imtafe, 1 ) ), flush=True, file=logfile )
                    np.save( expt_dir + "lct_ep" +str(epoch) + "_r" +str(i) + "_losses.npy", losses_vl )
            auc_epochs.append( auc_list )
            imtafe_epochs.append( imtafe_list )
            print_and_log( "---- --- ----" , flush=True, file=logfile )

            # saving the model
            if epoch % 10 == 0:
                print_and_log("saving out jetCLR model", flush=True, file=logfile)
                tms0 = time.time()
                torch.save(net.state_dict(), expt_dir + "model_ep" + str(epoch) + ".pt")
                tms1 = time.time()
                print_and_log( f"time taken to save model: {round( tms1-tms0, 1 )}s", flush=True, file=logfile )
            
            # saving out training stats
            if epoch % 10 == 0:
                print_and_log( "saving out data/results", flush=True, file=logfile )
                tds0 = time.time()
                np.save( expt_dir + "clr_losses.npy", losses )
                np.save( expt_dir + "auc_epochs.npy", np.array( auc_epochs ) )
                np.save( expt_dir + "imtafe_epochs.npy", np.array( imtafe_epochs ) )
                np.save( expt_dir + "align_loss_train.npy", loss_align_epochs )
                np.save( expt_dir + "uniform_loss_train.npy", loss_uniform_epochs )
                tds1 = time.time()
                print_and_log( f"time taken to save data: {round( tds1-tds0, 1 )}s", flush=True, file=logfile )

    t2 = time.time()

    print_and_log( "JETCLR TRAINING DONE, time taken: " + str( np.round( t2-t1, 2 ) ), flush=True, file=logfile )

    # save out results
    print_and_log( "saving out data/results", flush=True, file=logfile )
    np.save( expt_dir+"clr_losses.npy", losses )
    np.save( expt_dir+"auc_epochs.npy", np.array( auc_epochs ) )
    np.save( expt_dir+"imtafe_epochs.npy", np.array( imtafe_epochs ) )
    np.save( expt_dir+"align_loss_train.npy", loss_align_epochs )
    np.save( expt_dir+"uniform_loss_train.npy", loss_uniform_epochs )

    # save out final trained model
    print_and_log( "saving out final jetCLR model", flush=True, file=logfile )
    torch.save(net.state_dict(), expt_dir+"final_model.pt")

    print_and_log( "starting the final LCT run", flush=True, file=logfile )

    # evaluate the network on the testing data, applying some augmentations first if it's required
    if args.trs:
        vl_dat_1 = translate_jets( vl_dat_1, width=args.trsw )
        vl_dat_2 = translate_jets( vl_dat_2, width=args.trsw )
    with torch.no_grad():
        net.eval()
        #vl_reps_1 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
        #vl_reps_2 = F.normalize( net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu(), dim=-1 ).numpy()
        vl_reps_1 = net.forward_batchwise( torch.Tensor( vl_dat_1 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
        vl_reps_2 = net.forward_batchwise( torch.Tensor( vl_dat_2 ).transpose(1,2), args.batch_size, use_mask=args.mask, use_continuous_mask=args.cmask ).detach().cpu().numpy()
        net.train()

    # final LCT for each rep layer
    for i in range(vl_reps_1.shape[1]):
        t3 = time.time()
        out_dat_f, out_lbs_f, losses_f = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, vl_reps_1[:,i,:], vl_lab_1, vl_reps_2[:,i,:], vl_lab_2 )
        auc, imtafe = get_perf_stats( out_lbs_f, out_dat_f )
        ep=0
        step_size = 25
        for lss in losses_f[::step_size]:
            print_and_log( f"(rep layer {i}) epoch: " + str( ep ) + ", loss: " + str( lss ), flush=True, file=logfile )
            ep+=step_size
        print_and_log( f"(rep layer {i}) auc: "+str( round(auc, 4) ), flush=True, file=logfile )
        print_and_log( f"(rep layer {i}) imtafe: "+str( round(imtafe, 1) ), flush=True, file=logfile )
        t4 = time.time()
        np.save( expt_dir+f"linear_losses_{i}.npy", losses_f )
        np.save( expt_dir+f"test_linear_cl_{i}.npy", out_dat_f )

    print_and_log( "final LCT  done and output saved, time taken: " + str( np.round( t4-t3, 2 ) ), flush=True, file=logfile )
    print_and_log("............................", flush=True, file=logfile)

    t5 = time.time()

    print_and_log( "----------------------------", flush=True, file=logfile )
    print_and_log( "----------------------------", flush=True, file=logfile )
    print_and_log( "----------------------------", flush=True, file=logfile )
    print_and_log( "ALL DONE, total time taken: " + str( np.round( t5-t0, 2 ) ), flush=True, file=logfile )


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v2/toptagging/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--num-train-files",
        type=int,
        action="store",
        dest="num_train_files",
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
        default=4,
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
        type=bool,
        action="store",
        default=False,
        help="use mask in transformer",
    )
    parser.add_argument(
        "--cmask",
        type=bool,
        action="store",
        default=True,
        help="use continuous mask in transformer",
    )
    parser.add_argument(
        "--n-epoch", type=int, action="store", dest="n_epochs", default=300, help="Epochs"
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

    args = parser.parse_args()
    main(args)