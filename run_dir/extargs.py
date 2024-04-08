#!/bin/env python3
logfile = "/ssl-jet-vol-v2/JetCLR/logs/Top_Tagging/zz-simCLR-trial-log.txt"
tr_dat_path = "/ssl-jet-vol-v2/toptagging/train/processed/3_features_raw/data/data_0.pt"
tr_lab_path = "/ssl-jet-vol-v2/toptagging/train/processed/3_features_raw/labels/labels_0.pt"
nconstit = 50
model_dim = 8
output_dim = 8
n_heads = 4
dim_feedforward = 8
n_layers = 4
n_head_layers = 2
opt = "adam"
sbratio = 1.0
n_epochs = 500
learning_rate = 0.00005
batch_size = 256
temperature = 0.10
rot = True
ptd = True
ptcm = 0.1
ptst = 0.1
trs = True
trsw = 1.0
cf = True
mask= False
cmask = True
expt = "zz-simCLR-8-trial"

