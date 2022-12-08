
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.parallel
import pandas as pd
from sklearn.model_selection import train_test_split


######################################################################
# Load datasets (Signal)
######################################################################
def get_DeepMIMO_data(data_name='DeepMIMO_O1_3p5'):
    """
    Read and process DeepMIMO dataset

    Return:
        c = spatial coordinates (lon/lat)
        x = features at location
        y = target variable
    """

    # read the /csv file
    csv = '../data/{}'.format(data_name+".csv")
    df_total = pd.read_csv(csv)
    
    # process
    y = df_total.y.to_numpy().reshape(-1, 1)
    df_total.drop(['y'], axis=1, inplace=True)
    data_arr = df_total.to_numpy()
    
    c = data_arr[:, :2]
    X = data_arr[:, 2:]

    return torch.tensor(c).float(), torch.tensor(X).float(), torch.tensor(y).float()


def get_mndt_signal_data(data_name='mndt_nr_signal_pci1_cell', target_runs='target_1'):

    # Get file and col name
    Tech_Name = data_name.split("_")[1]

    if Tech_Name == "nr":
        PCI_ID = data_name.split("_")[3].removeprefix("pci")
        Beam_ID = data_name.split("_")[4]
        file_name = "NR-signal-PCI{}-total-reduced.csv".format(PCI_ID)
        col_name = "PCI{}_{}_mean".format(PCI_ID, Beam_ID)

    elif Tech_Name == "lte":
        PCI_ID = data_name.split("_")[3].removeprefix("pci")
        file_name = "LTE-signal-total-reduced.csv"
        col_name = "PCI{}_mean".format(PCI_ID)

    # Read the csv file from Google Drive
    csv = '../data/{}'.format(file_name)
    df_total = pd.read_csv(csv)

    # Read dataset {c, X, y}
    c = df_total[["Lon", "Lat"]].to_numpy()
    X = df_total["GPS Head"].to_numpy().reshape(-1, 1)
    y = df_total[col_name].to_numpy().reshape(-1, 1)
    valid_idx = np.where(~np.isnan(y))[0]

    # Get training set token
    train_token = []
    if "target" in target_runs:
        target_runs = np.array(target_runs.split("_")[1:]).astype(int)
        for i in range(len(df_total)):
            runs = np.unique(df_total["run"][i].replace('[', "").replace(']', "").replace('\'', "").split(', ')).astype(int)
            if any(np.isin(target_runs, runs)):
                train_token.append(1)
            else:
                train_token.append(0)
        train_token = np.array(train_token).reshape(-1, 1)[valid_idx]

    return torch.tensor(c[valid_idx]).float(), torch.tensor(X[valid_idx]).float(), torch.tensor(y[valid_idx]).float(), train_token


def get_mndt_cqi_data(data_name='mndt_nr_cqi_pci1', target_runs='target_1'):

    # Get file and col name
    Tech_Name = data_name.split("_")[1]

    if Tech_Name == "nr":
        PCI_ID = data_name.split("_")[3].removeprefix("pci")
        file_name = "NR-cqi-PCI{}-total-reduced.csv".format(PCI_ID)
        col_y_name = "CQI_mean"
        col_x_name = ["GPS Head", "ssRSRP_mean", "ssRSRQ_mean", "ssSINR_mean", "ssbRSRP_mean", "criRSRP_mean", "Pathloss_mean", "BLER_mean", "CQI_mean"]

    elif Tech_Name == "lte":
        PCI_ID = data_name.split("_")[3].removeprefix("pci")
        file_name = "LTE-cqi-PCI{}-total-reduced.csv".format(PCI_ID)
        col_y_name = "CQI_mean"
        col_x_name = ["GPS Head", "RSRP_mean", "RSRQ_mean", "RSSI_mean", "SINR_mean", "BLER_mean", "Pathloss_mean"]

    # Read the csv file from Google Drive
    csv = '../data/{}'.format(file_name)
    df_total = pd.read_csv(csv)

    # Read dataset {c, X, y}
    c = df_total[["Lon", "Lat"]].to_numpy()
    X = df_total[col_x_name].to_numpy().reshape(-1, len(col_x_name))
    y = df_total[col_y_name].to_numpy().reshape(-1, 1)
    valid_idx = np.where(~np.isnan(y))[0]

    # Get training set token
    train_token = []
    if "target" in target_runs:
        target_runs = np.array(target_runs.split("_")[1:]).astype(int)
        for i in range(len(df_total)):
            runs = np.unique(df_total["run"][i].replace('[', "").replace(']', "").replace('\'', "").split(', ')).astype(int)
            if any(np.isin(target_runs, runs)):
                train_token.append(1)
            else:
                train_token.append(0)
        train_token = np.array(train_token).reshape(-1, 1)[valid_idx]

    return torch.tensor(c[valid_idx]).float(), torch.tensor(X[valid_idx]).float(), torch.tensor(y[valid_idx]).float(), train_token


######################################################################
# Split dataset
######################################################################
def get_split_idx(total_c, config):
    """
    Split the dataset into train and test set
    """

    # Method 1: Dataset contains training idx
    if "target" in config["split_method"]:
        train_idx = np.where(config["train_token"] == 1)[0]
        test_idx = np.where(config["train_token"] == 0)[0]
        if config["shuffle_split"]:
            np.random.shuffle(train_idx)
        valid_idx = train_idx[:int(config["valid_ratio"] * len(train_idx))]
        train_idx = train_idx[int(config["valid_ratio"] * len(train_idx)):]

    # Method 2: Random split the dataset into train and test
    elif config["split_method"] == "random":
        train_idx, test_idx = train_test_split(np.arange(total_c.shape[0]), train_size=config["train_ratio"], shuffle=config["shuffle_split"],)
        if config["shuffle_split"]:
            np.random.shuffle(train_idx)
        valid_idx = train_idx[:int(config["valid_ratio"]*len(train_idx))]
        train_idx = train_idx[int(config["valid_ratio"]*len(train_idx)):]

    return train_idx, valid_idx, test_idx


######################################################################
# MyDataset
######################################################################
class MyDataset(Dataset):
    def __init__(self, c, X, y):
        self.c = c
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        c_i = torch.tensor(self.c[idx])
        X_i = torch.tensor(self.X[idx])
        y_i = torch.tensor(self.y[idx])
        return c_i, X_i, y_i

class MyDataset2(Dataset):
    def __init__(self, c, A, X, y):
        self.c = c
        self.A = A
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        c_i = torch.tensor(self.c[idx])
        A_i = torch.tensor(self.A[idx])
        X_i = torch.tensor(self.X[idx])
        y_i = torch.tensor(self.y[idx])
        return c_i, A_i, X_i, y_i
