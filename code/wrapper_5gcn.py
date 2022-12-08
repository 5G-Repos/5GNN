
import torch
import numpy as np
import pandas as pd
from utils_data import MyDataset, MyDataset2
from utils_plot import draw_data_points
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer
from utils_metric import rmse, mae
from torch_geometric.nn import knn_graph
from utils_spatial import makeEdgeWeight
from sklearn.metrics import pairwise


class my5GCN:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = config["opt"](self.model.parameters(), **config["opt_params"])
        self.tasks = config["loss_tasks"]
        self.df_metrics = pd.DataFrame(index=range(self.config["epochs"]), columns=["train_loss", "test_loss"])

        self.obs_c = None
        self.obs_X = None
        self.obs_y = None

        if self.config["num_nbrs"]:
            self.KNN_train = NearestNeighbors(n_neighbors=1+self.config["num_nbrs"])
            self.KNN_test = NearestNeighbors(n_neighbors=self.config["num_nbrs"])

    def loss_wrapper(self, true, pred):
        total_loss = []
        if "rmse" in self.tasks.keys():
            loss_rmse = rmse(true, pred)
            total_loss.append(loss_rmse)
        total_loss = sum(total_loss)
        return total_loss

    def sample_graph(self, idx_ce, idx_other, batch_c, batch_X):
        ci = torch.cat((batch_c[[idx_ce]], self.obs_c[idx_other]), dim=0)
        Ai = pairwise.rbf_kernel(ci)

        # Xi_ce = np.array([batch_X[idx_ce].tolist() + [1] + [0]])
        Xi_ce = np.array([batch_c[idx_ce].tolist() + [0]*(batch_X.shape[1]-batch_c.shape[1]) + [1] + [0]])

        Xi_nbr = self.obs_data[idx_other]
        Xi = np.concatenate((Xi_ce, Xi_nbr), axis=0)
        return Ai, Xi

    def get_nbr_feat(self, mode, batch_c, batch_X):
        # Find neighbors for the incoming data based on the observed data
        if mode == "train":
            nbr_dis, nbr_idx = self.KNN_train.kneighbors(X=batch_c, return_distance=True)  # distance: close to far
            nbr_idx = nbr_idx[:, 1:]                                               # we don't want self-self distance
        else:
            nbr_idx = self.KNN_test.kneighbors(X=batch_c, return_distance=False)

        # Sample local graph
        A_list = []
        X_list = []
        for i in range(nbr_idx.shape[0]):
            # Sample local knn graph
            knn_idx = nbr_idx[i, :self.config["num_nbrs"]]
            Ai, Xi = self.sample_graph(i, knn_idx, batch_c, batch_X)
            A_list.append(Ai)
            X_list.append(Xi)
        A_total = torch.Tensor(A_list)
        X_total = torch.Tensor(X_list)

        return A_total, X_total

    def collate_fn(self, data_list):
        # Default function
        batch_c, batch_A, batch_X, batch_y = list(zip(*data_list))
        batch_c = torch.stack(batch_c)
        batch_A = torch.stack(batch_A)
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y)

        # Prepare data in dataloader
        edge_index = knn_graph(batch_c, k=self.config["num_nbrs"])  # +1, loop=True
        edge_weight = makeEdgeWeight(batch_c, edge_index)
        edge_ones = torch.ones(len(edge_index[0])) / self.config["num_nbrs"]
        return batch_c, batch_X, batch_y, edge_index, edge_weight, edge_ones, batch_A

    def train(self, train_c: torch.Tensor, train_X: torch.Tensor, train_y: torch.Tensor,
                    valid_c: torch.Tensor, valid_X: torch.Tensor, valid_y: torch.Tensor,
                    test_c: torch.Tensor, test_X: torch.Tensor, test_y: torch.Tensor):
        """
        Main training function
        :param: dtype=torch.float32
        """
        print("Start Training: ")

        # Load observed points
        self.obs_c = train_c
        self.obs_X = train_X
        self.obs_y = train_y
        self.obs_data = np.concatenate((self.obs_X, np.zeros((len(self.obs_X), 1)), self.obs_y), axis=1)

        # Init KNN method if using nbr's info
        if self.config["num_nbrs"]:
            self.KNN_train.fit(self.obs_c)
            self.KNN_test.fit(self.obs_c)

        # Process Nbr infomation
        A, H = self.get_nbr_feat(mode="train", batch_c=train_c, batch_X=train_X)

        # Prepare dataloader
        train_dataset = MyDataset2(c=train_c, A=A, X=H, y=train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle_loader"], collate_fn=self.collate_fn, num_workers=2)

        # Training
        for epoch in range(self.config["epochs"]):
            batch_train_loss_list = []
            epoch_start = timer()
            batch_data_time = 0

            for batch in train_loader:
                self.model.train()
                batch_data_start = timer()

                # Prepare batched data
                batch_c, batch_H, batch_y, edge_index, edge_weight, edge_ones, batch_A = batch
                batch_A = batch_A.to(self.config["device"])
                batch_H = batch_H.to(self.config["device"])
                batch_y = batch_y.to(self.config["device"])
                edge_index = edge_index.to(self.config["device"])
                edge_weight = edge_weight.to(self.config["device"])
                edge_ones = edge_ones.to(self.config["device"])
                batch_data_time += timer() - batch_data_start

                # Train
                self.model.zero_grad()
                out = self.model(batch_A, batch_H, edge_index, edge_weight)
                loss = self.loss_wrapper(batch_y, out)
                loss.backward()
                self.optimizer.step()

                # Save batch log
                batch_train_loss_list.append(loss.item())
            epoch_duration = timer()-epoch_start
            epoch_train_loss = np.mean(batch_train_loss_list)
            self.df_metrics.loc[epoch, ["train_loss"]] = epoch_train_loss
            print("Epoch [{}/{}], avg_train_loss {:.6f}, epoch_duration {:.3f}s, data_prepare {:.3f}s"
                  .format(epoch, self.config["epochs"]-1, epoch_train_loss, epoch_duration, batch_data_time), end="\r")

            # Evaluate and save checkpoints
            if 30 <= epoch <= 150:
                ck_intv = 3
            else:
                ck_intv = self.config["checkpoint_interval"]

            if epoch % ck_intv == 0 or epoch == self.config["epochs"]-1:
                self.model.eval()

                valid_loss, valid_metric_dict = self.eval_model(test_c=valid_c, test_X=valid_X, test_y=valid_y)
                self.df_metrics.loc[epoch, ["valid_loss"]] = valid_loss
                self.df_metrics.loc[epoch, ["valid_metric_rmse"]] = valid_metric_dict["rmse"]
                self.df_metrics.loc[epoch, ["valid_metric_mae"]] = valid_metric_dict["mae"]

                test_loss, test_metric_dict = self.eval_model(test_c=test_c, test_X=test_X, test_y=test_y)
                self.df_metrics.loc[epoch, ["test_loss"]] = test_loss
                self.df_metrics.loc[epoch, ["test_metric_rmse"]] = test_metric_dict["rmse"]
                self.df_metrics.loc[epoch, ["test_metric_mae"]] = test_metric_dict["mae"]
                print("\nEpoch [{}/{}], valid_loss {:.4f}, valid_rmse {:.4f}, valid_mae {:.4f}, test_loss {:.4f}, test_rmse {:.4f}, test_mae {:.4f}"
                      .format(epoch, self.config["epochs"]-1, valid_loss, valid_metric_dict["rmse"], valid_metric_dict["mae"],
                                                              test_loss, test_metric_dict["rmse"], test_metric_dict["mae"]))

                if self.config["save_checkpoint"]:
                    self.checkpoint_model(epoch)
                if self.config["generate_image"]:
                    self.generate_image(epoch, test_c, test_X)

    def checkpoint_model(self, epoch: int):
        # save mlp model
        # e.g., ./checkpoint_0417-2322/mlp_100.pkl.gz
        torch.save(self.model, "{}mlp_{}.pkl.gz".format(self.config["checkpoint_path"], epoch))
        self.df_metrics.to_csv(self.config["checkpoint_path"] + "metrics.csv")

    def eval_model(self, test_c: torch.Tensor, test_X: torch.Tensor, test_y: torch.Tensor):
        # Inference
        out = self.predict(test_c, test_X)
        loss = self.loss_wrapper(test_y, out).item()

        # Calculate metrics
        test_y_pred = out
        test_y_true = test_y

        all_y_pred = torch.cat((self.obs_y, test_y_pred), dim=0)
        all_y_true = torch.cat((self.obs_y, test_y_true), dim=0)
        all_c = torch.cat((self.obs_c, test_c), dim=0)

        metric_dict = {}
        metric_dict['rmse'] = rmse(test_y_true, test_y_pred).item()
        metric_dict['mae'] = mae(test_y_true, test_y_pred).item()
        metric_dict['mie'] = 0  # mie(all_y_true, all_y_pred, all_c).item()
        metric_dict['bp_mie'] = 0  # bp_mie(all_y_true, all_y_pred, all_c).item()

        # print("\ntime {:.6f}, num_samples {:.6f}".format(timer() - time_start, len(test_y_pred)))
        return loss, metric_dict

    def predict(self, test_c, test_X):
        # Prepare data
        A, H = self.get_nbr_feat(mode="test", batch_c=test_c, batch_X=test_X)
        A = A.to(self.config["device"])
        H = H.to(self.config["device"])

        # KNN graph for the whole test dataset
        edge_index = knn_graph(test_c, k=self.config["num_nbrs"])
        edge_weight = makeEdgeWeight(test_c, edge_index)
        edge_index = edge_index.to(self.config["device"])
        edge_weight = edge_weight.to(self.config["device"])

        # Inference
        out = self.model(A,H, edge_index, edge_weight)
        out = out.detach().cpu()
        return out

    def generate_image(self, epoch: int, test_c: torch.Tensor, test_X: torch.Tensor):
        # Inference
        test_y_pred = self.predict(test_c, test_X)

        # Plot and save charts
        total_out = torch.cat((self.obs_y, test_y_pred), dim=0).numpy()
        draw_data_points(epoch=epoch, total_y=total_out, train_c=self.obs_c, test_c=test_c, config=self.config)

