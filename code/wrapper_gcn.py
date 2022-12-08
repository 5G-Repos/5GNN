import torch
import numpy as np
import pandas as pd
from utils_data import MyDataset
from utils_plot import draw_data_points
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from torch_geometric.nn import knn_graph
from utils_spatial import makeEdgeWeight
from utils_metric import rmse, mae


# Wrapper
class myGCN:
    def __init__(self, model, config):
        self.GCN = model
        self.config = config
        self.optimizer = config["opt"](self.GCN.parameters(), **config["opt_params"])
        self.tasks = config["loss_tasks"]
        self.df_metrics = pd.DataFrame(index=range(self.config["epochs"]), columns=["train_loss", "test_loss"])

        self.obs_c = None
        self.obs_X = None
        self.obs_y = None

    def loss_wrapper(self, true, pred):
        total_loss = []
        if "rmse" in self.tasks.keys():
            loss_rmse = rmse(true, pred)
            total_loss.append(loss_rmse)
        total_loss = sum(total_loss)
        return total_loss

    def collate_fn(self, data_list):
        # Default function
        batch_c, batch_X, batch_y = list(zip(*data_list))
        batch_c = torch.stack(batch_c)
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y)

        # Prepare data in dataloader
        edge_index = knn_graph(batch_c, k=self.config["num_nbrs"])  # +1, loop=True
        if self.config["make_weight_gb"]:
            edge_weight = makeEdgeWeight(batch_c, edge_index)
        else:
            edge_weight = torch.ones(len(edge_index[0]))  # / self.config["num_nbrs"]

        # Semi supervised learning
        batch_size = batch_y.shape[0]
        feature_dim = batch_X.shape[1]
        mask_id = torch.tensor(np.random.choice(np.arange(0, batch_size), int(0.5 * batch_size), replace=False), dtype=int)
        mask = torch.ones(batch_size, feature_dim + 1)
        mask[mask_id, 2:] = 0
        batch_X = torch.cat((batch_X, batch_y), dim=1) * mask

        return batch_c, batch_X, batch_y, edge_index, edge_weight, mask_id

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

        # Prepare dataloader
        train_dataset = MyDataset(c=train_c, X=train_X, y=train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=self.config["shuffle_loader"], collate_fn=self.collate_fn, num_workers=4)

        # Training loop
        for epoch in range(self.config["epochs"]):
            batch_train_loss_list = []
            epoch_start = timer()
            batch_data_time = 0

            for batch in train_loader:
                self.GCN.train()
                batch_data_start = timer()

                # Prepare batched data
                batch_c, batch_X, batch_y, edge_index, edge_weight, mask_id = batch

                batch_X = batch_X.to(self.config["device"])
                batch_y = batch_y.to(self.config["device"])
                edge_index = edge_index.to(self.config["device"])
                edge_weight = edge_weight.to(self.config["device"])

                batch_data_time += timer() - batch_data_start

                # Train
                self.GCN.zero_grad()
                out = self.GCN(batch_X, edge_index, edge_weight)
                loss = self.loss_wrapper(batch_y[mask_id], out[mask_id])
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
            if 50 <= epoch <= 150:
                ck_intv = 10
            else:
                ck_intv = self.config["checkpoint_interval"]

            if epoch % ck_intv == 0 or epoch == self.config["epochs"] - 1:
                self.GCN.eval()

                valid_loss, valid_metric_dict = self.eval_model(test_c=valid_c, test_X=valid_X, test_y=valid_y)
                self.df_metrics.loc[epoch, ["valid_loss"]] = valid_loss
                self.df_metrics.loc[epoch, ["valid_metric_rmse"]] = valid_metric_dict["rmse"]
                self.df_metrics.loc[epoch, ["valid_metric_mae"]] = valid_metric_dict["mae"]

                test_loss, test_metric_dict = self.eval_model(test_c=test_c, test_X=test_X, test_y=test_y)
                self.df_metrics.loc[epoch, ["test_loss"]] = test_loss
                self.df_metrics.loc[epoch, ["test_metric_rmse"]] = test_metric_dict["rmse"]
                self.df_metrics.loc[epoch, ["test_metric_mae"]] = test_metric_dict["mae"]

                print("\nEpoch [{}/{}], valid_loss {:.4f}, valid_rmse {:.4f}, valid_mae {:.4f}, test_loss {:.4f}, test_rmse {:.4f}, test_mae {:.4f}"
                      .format(epoch, self.config["epochs"] - 1, valid_loss, valid_metric_dict["rmse"], valid_metric_dict["mae"],
                                                                test_loss, test_metric_dict["rmse"], test_metric_dict["mae"]))

                if self.config["save_checkpoint"]:
                    self.checkpoint_model(epoch)
                if self.config["generate_image"]:
                    self.generate_image(epoch, test_c, test_X)

    def checkpoint_model(self, epoch: int):
        # save mlp model
        # e.g., ./checkpoint_0417-2322/mlp_100.pkl.gz
        torch.save(self.GCN, "{}gcn_{}.pkl.gz".format(self.config["checkpoint_path"], epoch))
        self.df_metrics.to_csv(self.config["checkpoint_path"] + "metrics.csv")

    def eval_model(self, test_c: torch.Tensor, test_X: torch.Tensor, test_y: torch.Tensor):
        # Prepare batch data
        batch_size = self.obs_X.shape[0] + test_y.shape[0]
        feature_dim = self.obs_X.shape[1]
        mask_id = torch.tensor(np.arange(self.obs_X.shape[0], batch_size), dtype=int)
        mask = torch.ones(batch_size, feature_dim + 1)
        mask[mask_id, 2:] = 0

        batch_c = torch.cat((self.obs_c, test_c), dim=0)
        batch_X = torch.cat((self.obs_X, test_X), dim=0)
        batch_y = torch.cat((self.obs_y, test_y), dim=0)
        batch_X = torch.cat((batch_X, batch_y), dim=1) * mask

        # Inference
        out = self.predict(batch_c, batch_X)
        loss = self.loss_wrapper(batch_y[mask_id], out[mask_id]).item()

        # Calculate metrics
        batch_y_pred = out
        batch_y_true = batch_y

        metric_dict = {}
        metric_dict['rmse'] = rmse(batch_y_true[mask_id], batch_y_pred[mask_id]).item()
        metric_dict['mae'] = mae(batch_y_true[mask_id], batch_y_pred[mask_id]).item()
        return loss, metric_dict

    def predict(self, test_c, test_X):
        # Prepare data
        test_X = test_X.to(self.config["device"])
        # KNN graph for the whole test dataset
        edge_index = knn_graph(test_c, k=self.config["num_nbrs"])
        edge_weight = makeEdgeWeight(test_c, edge_index)
        edge_index = edge_index.to(self.config["device"])
        edge_weight = edge_weight.to(self.config["device"])

        # Inference
        out = self.GCN(test_X, edge_index, edge_weight)
        out = out.detach().cpu()
        return out

    def generate_image(self, epoch: int, test_c: torch.Tensor, test_X: torch.Tensor):
        # Inference
        test_y_pred = self.predict(test_c, test_X)

        # Plot and save charts
        total_out = torch.cat((self.obs_y, test_y_pred), dim=0).numpy()
        draw_data_points(epoch=epoch, total_y=total_out, train_c=self.obs_c, test_c=test_c, config=self.config)
