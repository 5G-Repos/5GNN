
import torch
import pandas as pd
import numpy as np
from utils_plot import draw_data_points
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from utils_metric import rmse, mae
from timeit import default_timer as timer


class myKriging:
    def __init__(self, config):
        self.config = config
        self.df_metrics = pd.DataFrame(index=range(1), columns=["train_loss", "test_loss", "metric_rmse"])

        self.Kriging = None
        self.obs_c = None
        self.obs_X = None
        self.obs_y = None

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

        # Training
        time_start = timer()

        if self.config["model_name"] == 'ok':
            self.Kriging = OrdinaryKriging(np.array(self.obs_c[:, 0]),
                                           np.array(self.obs_c[:, 1]),
                                           np.array(self.obs_y),)
        elif self.config["model_name"] == 'uk':
            self.Kriging = UniversalKriging(np.array(self.obs_c[:, 0]),
                                            np.array(self.obs_c[:, 1]),
                                            np.array(self.obs_y),)

        # Evaluate
        metric_dict = self.eval_model(test_c=test_c, test_X=test_X, test_y=test_y)
        self.df_metrics.loc[0, ["metric_rmse"]] = metric_dict["rmse"]
        self.df_metrics.loc[0, ["metric_mae"]] = metric_dict["mae"]
        print("\ntime {:.6f}, test_rmse {:.6f}, test_mae {:.6f}"
              .format(timer() - time_start, metric_dict["rmse"], metric_dict["mae"]))

        if self.config["generate_image"]:
            self.generate_image(valid_c, valid_X, test_c, test_X)

    def eval_model(self, test_c: torch.Tensor, test_X: torch.Tensor, test_y: torch.Tensor):
        # Inference
        if self.config["model_name"] == 'ok' or self.config["model_name"] == 'uk':
            out = self.predict(test_c)
            test_y_pred = torch.tensor(out)
        elif self.config["model_name"] == 'rk':
            out = self.predict(test_c, test_X)
            test_y_pred = torch.tensor(out.reshape(-1, 1))

        # Calculate metrics
        test_y_true = test_y

        all_y_pred = torch.cat((self.obs_y, test_y_pred), dim=0)
        all_y_true = torch.cat((self.obs_y, test_y_true), dim=0)
        all_c = torch.cat((self.obs_c, test_c), dim=0)

        metric_dict = {}
        metric_dict['rmse'] = rmse(test_y_true, test_y_pred).item()
        metric_dict['mae'] = mae(test_y_true, test_y_pred).item()
        return metric_dict

    def predict(self, test_c, test_X=None):
        if self.config["model_name"] == 'ok' or self.config["model_name"] == 'uk':
            test_c = np.array(test_c)
            Z_values, sigmasq = self.Kriging.execute("points", test_c[:, 0], test_c[:, 1])
            out = Z_values.data.reshape(-1, 1)
        elif self.config["model_name"] == 'rk':
            test_c = np.array(test_c)
            out = self.Kriging.predict(test_X, test_c)
        return out

    def generate_image(self, valid_c: torch.Tensor, valid_X: torch.Tensor, test_c: torch.Tensor, test_X: torch.Tensor):
        # Prepare data
        total_X = np.array(torch.cat((self.obs_X, valid_X, test_X), dim=0))
        total_c = np.array(torch.cat((self.obs_c, valid_c, test_c), dim=0))

        # Inference
        out = self.predict(total_c)
        total_out = out

        # Plot and save charts
        draw_data_points(epoch=0, total_y=total_out, train_c=self.obs_c, valid_c=valid_c, test_c=test_c, config=self.config)

