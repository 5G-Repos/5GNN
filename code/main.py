
import os
import argparse
import logging
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils_data import get_DeepMIMO_data, get_mndt_signal_data, get_mndt_cqi_data
from utils_data import get_split_idx
from utils_metric import report_best_metric
from utils_plot import plot_loss_curve, plot_metric_curve, draw_imputation_map
from utils_plot import RedToGreen

from model_gcn import GCN, GraphSAGE, GAT, GIN, GCNII, EGC, SuperGAT
from model_5gcn import GCN5, GraphSAGE5

from wrapper_kriging import myKriging
from wrapper_gcn import myGCN
from wrapper_5gcn import my5GCN


import warnings
warnings.simplefilter("ignore")


##########################################################
# Hyper-parameters
##########################################################
parser = argparse.ArgumentParser(description='5GNN')

# System
parser.add_argument('-r', '--random_state', type=int, default=42)
parser.add_argument('-gpu_id', '--gpu_id', type=str,
                    default='0', choices=['0', '1', '2'])

# Data
parser.add_argument('-data_name', '--data_name', type=str,
                    default='mndt_nr_cqi_pci1', choices=['DeepMIMO_O1_3p5', 'DeepMIMO_O1_3p5B', 'DeepMIMO_O1_28', 'DeepMIMO_O1_28B',
                                                        'mndt_lte_signal_pci28813', 'mndt_lte_signal_pci28866', 'mndt_lte_signal_pci22466', 'mndt_nr_signal_pci1_cell',
                                                        'mndt_lte_cqi_pci28813', 'mndt_lte_cqi_pci28866', 'mndt_lte_cqi_pci22466', 'mndt_nr_cqi_pci1'])
parser.add_argument('-split_method', '--split_method', type=str,
                    default="random", choices=["random", "target_14_7"])
parser.add_argument('-train_ratio', '--train_ratio', type=float, default=0.5)
parser.add_argument('-valid_ratio', '--valid_ratio', type=float, default=0.5)
parser.add_argument('-shuffle_split', '--shuffle_split', type=bool, default=True)
parser.add_argument('-shuffle_loader', '--shuffle_loader', type=bool, default=True)

# Model
parser.add_argument('-model_name', '--model_name', type=str,
                    default='ok', choices=['ok', 'uk',
                                           'gcn', 'graphsage', 'gat', 'gin', 'gcn2', 'egc', 'supergat',
                                           '5gnn'])
parser.add_argument('-use_neighbours', '--use_neighbours', type=bool, default=True)
parser.add_argument('-use_coord', '--use_coord', type=bool, default=True)
parser.add_argument('-use_feat', '--use_feat', type=bool, default=True)
parser.add_argument('-use_label', '--use_label', type=bool, default=True)
parser.add_argument('-num_neighbours', '--num_neighbours', type=float, default=10)

# Training
parser.add_argument('-epochs', '--epochs', type=int, default=250)
parser.add_argument('-batch_size', '--batch_size', type=int, default=1024)
parser.add_argument('-lr', '--lr', type=float, default=0.003)

# Checkpoints
parser.add_argument('-save_checkpoint', '--save_checkpoint', type=bool, default=False)
parser.add_argument('-checkpoint_interval', '--checkpoint_interval', type=int, default=10)
parser.add_argument('-generate_image', '--generate_image', type=bool, default=False)

# Parse args
args = parser.parse_args()


##########################################################
# Set environment
##########################################################
# Random
np.random.seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)

# Device
device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')

# Checkpoint
checkpoint_path = "./checkpoint_{}{}-{}{}-{}/".format(datetime.now().strftime("%m"), datetime.now().strftime("%d"),
                                                      datetime.now().strftime("%H"), datetime.now().strftime("%M"),
                                                      datetime.now().strftime("%S"))
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(checkpoint_path+'log.txt'))
print = logger.info

# Config
config = {"device": device,                           # Hardware device: GPU or CPU
          "point_size": 1,                            # Size of scatter points in the plots

          "data_name": args.data_name,                # Dataset name
          "split_method": args.split_method,          # Split methods
          "shuffle_split": args.shuffle_split,        # Shuffle the dataset while splitting
          "shuffle_loader": args.shuffle_loader,      # Shuffle the dataloader while training
          "train_ratio": args.train_ratio,            # Ratio of training set
          "valid_ratio": args.valid_ratio,            # Ratio of validation set
          "scale_c": MinMaxScaler(),                  # Sklearn scaling method
          "scale_x": MinMaxScaler(),                  # Sklearn scaling method
          "scale_y": MinMaxScaler(),                  # Sklearn scaling method (* important for SOTA comparison)

          "model_name": args.model_name,              # Machine model name
          "epochs": args.epochs,                      # Training epochs
          "batch_size": args.batch_size,              # Training batch size
          "lr": args.lr,                              # Learning rate
          "use_feat": args.use_feat,                  # Use feature (i.e., regression or imputation)
          "use_coord": args.use_coord,                # Use coordinate as part of features
          "use_label": args.use_label,                # Use label as part of features (i.e., label propagation)
          "num_nbrs": args.num_neighbours,            # Number of neighbors considered
          "make_weight_gb": False,                    # Make the weight for the global knn graph

          "save_checkpoint": args.save_checkpoint,
          "checkpoint_interval": args.checkpoint_interval,
          "checkpoint_path": checkpoint_path,
          "generate_image": args.generate_image, }

# Update config and check consistency
if not args.use_neighbours:
    config["num_nbrs"] = 0


##########################################################
# Load data
##########################################################
# Load dataset
if 'DeepMIMO' in config["data_name"]:
    total_c, total_X, total_y = get_DeepMIMO_data(config["data_name"])
if 'mndt' in config["data_name"] and 'signal' in config["data_name"]:
    total_c, total_X, total_y, train_token = get_mndt_signal_data(config["data_name"], config["split_method"])
    config["train_token"] = train_token
if 'mndt' in config["data_name"] and 'cqi' in config["data_name"]:
    total_c, total_X, total_y, train_token = get_mndt_cqi_data(config["data_name"])
    config["train_token"] = train_token

# Visialize dataset
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
for i, label in enumerate(range(total_y.shape[1])):
    norm_y = (total_y[:, i]-min(total_y[:, i]))/(max(total_y[:, i])-min(total_y[:, i]))
    ax1.scatter(total_c[:, 0], total_c[:, 1], c=norm_y, cmap=RedToGreen, marker='o', s=config["point_size"], vmin=0.3, vmax=0.9)
    ax1.set_xlabel(r'$c^{(1)}$', fontsize=14)
    ax1.set_ylabel(r'$c^{(2)}$', fontsize=14)
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_title("Dataset")
    fig.savefig(config["checkpoint_path"]+"dataset_"+str(label)+".png", dpi=100, transparent=True, bbox_inches="tight")
    # fig.show()


##########################################################
# Process data
##########################################################
# Process features and update config
if not config["use_feat"]:
    # If not use the feature, then dummy it.
    total_X = torch.ones(total_X.shape[0], 1).size()

if config["use_coord"]:
    # If use the coord, then regard it as feature
    total_X = torch.cat((total_c, total_X), dim=1)

if config["model_name"] in ['gcn', 'graphsage', 'gat', 'gin', 'gcn2', 'egc', 'supergat']:
    config["input_dim"] = total_X.shape[1]
    if total_c.shape[1] <= total_X.shape[1]-1:
        config["input_dim"] += 1  # 1
elif config["model_name"] == "5gnn":
    # config["input_dim"] = total_c.shape[1] + config["num_nbrs"] * (total_X.shape[1] + total_y.shape[1])
    config["input_dim"] = total_X.shape[1] + 1 + total_y.shape[1]
else:
    config["input_dim"] = total_X.shape[1] + config["num_nbrs"] * (total_X.shape[1] + total_y.shape[1])
config["output_dim"] = total_y.shape[1]


# Split dataset
train_idx, valid_idx, test_idx = get_split_idx(total_c, config)
train_c, valid_c, test_c = total_c[train_idx], total_c[valid_idx], total_c[test_idx]
train_X, valid_X, test_X = total_X[train_idx], total_X[valid_idx], total_X[test_idx]
train_y, valid_y, test_y = total_y[train_idx], total_y[valid_idx], total_y[test_idx]


# Scaling dataset
config["scale_c"].fit(train_c)
train_c = torch.tensor(config["scale_c"].transform(train_c))
valid_c = torch.tensor(config["scale_c"].transform(valid_c))
test_c = torch.tensor(config["scale_c"].transform(test_c))
config["scale_x"].fit(train_X)
train_X = torch.tensor(config["scale_x"].transform(train_X))
valid_X = torch.tensor(config["scale_x"].transform(valid_X))
test_X = torch.tensor(config["scale_x"].transform(test_X))
config["scale_y"].fit(train_y)
train_y = torch.tensor(config["scale_y"].transform(train_y))
valid_y = torch.tensor(config["scale_y"].transform(valid_y))
test_y = torch.tensor(config["scale_y"].transform(test_y))


# Print dataset information
print("Dataset Information: ")
train_ratio = len(train_idx)/(len(train_idx)+len(valid_idx)+len(test_idx))
print("\tData_Name: {}\n\tNum_Samples: {}\n\tTrain_Ratio: {:.3f}\n\tShuffle_Split: {}\n\tNum_Features: {}\n"
      "\tNum_Neighbours: {}\n\tUse_Features: {}\n\tUse_Coordinate: {}"
      .format(config["data_name"], total_X.shape[0], train_ratio, config["shuffle_split"], total_X.shape[1],
              config["num_nbrs"], config["use_feat"], config["use_coord"]))


##########################################################
# Model training
##########################################################
# Print training information
print("Training Information: ")
print("\tModel_Name: {}\n\tInput_Dim: {}\n\tOutput_Dim: {}\n\tNum_Epochs: {}\n\tLearning_Rate: {}\n"
      "\tBatch_Size: {}\n\tShuffle_Loader: {}"
      .format(config["model_name"], config["input_dim"], config["output_dim"], config["epochs"], config["lr"],
              config["batch_size"], config["shuffle_loader"]))

# Pre-instantiation
if config["model_name"] in ['ok', 'uk']:
    # Model class
    myModel = myKriging(config)

elif config["model_name"] in ['gcn', 'graphsage', 'gat', 'gin', 'gcn2', 'egc', 'supergat', 'pegcn']:
    # GCNs
    if config["model_name"] == "gcn":
        model = GCN(config).to(device)            # ICLR 2017
    elif config["model_name"] == "graphsage":
        model = GraphSAGE(config).to(device)      # NeurIPS 2017
    elif config["model_name"] == "gat":
        model = GAT(config).to(device)            # ICLR 2018
    elif config["model_name"] == "supergat":
        model = SuperGAT(config).to(device)       # ICLR 2021
    elif config["model_name"] == "gin":
        model = GIN(config).to(device)            # ICLR 2019
    elif config["model_name"] == "gcn2":
        model = GCNII(config).to(device)          # ICML 2020
    elif config["model_name"] == "egc":
        model = EGC(config).to(device)            # ICLR 2022
    config["opt"] = torch.optim.Adam
    config["opt_params"] = {"lr": config["lr"]}
    # Model class
    config["loss_tasks"] = {"rmse": 1}
    myModel = myGCN(model, config)

elif config["model_name"] == "5gnn":
    # Ours
    model = GCN5(config).to(device)
    # model = GraphSAGE5(config).to(device)

    config["opt"] = torch.optim.Adam
    config["opt_params"] = {"lr": config["lr"]}
    # Model class
    config["loss_tasks"] = {"rmse": 1}
    myModel = my5GCN(model, config)


# Training [input_dtype = torch.float32]
myModel.train(train_c=train_c.float(), train_X=train_X.float(), train_y=train_y.float(),
              valid_c=valid_c.float(), valid_X=valid_X.float(), valid_y=valid_y.float(),
              test_c=test_c.float(), test_X=test_X.float(), test_y=test_y.float())


# Report the best metric
report_best_metric(myModel, config)


# Generate training loss plot
plot_loss_curve(myModel, config)
plot_metric_curve(myModel, config)
# draw_imputation_map(model=myModel, num_grid=75, total_C=total_c, train_C=train_c, config=config)

# Finish
print("Done!")

