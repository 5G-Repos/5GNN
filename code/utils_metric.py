
import numpy as np
import torch


def report_best_metric(myModel, config):
    metric_mins = {}
    if len(myModel.df_metrics) == 1:
        myModel.df_metrics.to_csv(config["checkpoint_path"] + "metric_report(one epoch only).csv")
        return
    if np.all(np.isnan(myModel.df_metrics.values.astype(float))):
        print("Skip the report generation since no valid values.")
        return
    else:
        idx = myModel.df_metrics.astype(float).idxmin()['valid_loss']
        df = myModel.df_metrics.iloc[idx, :]
        print("train_loss {:.4f}, valid_loss {:.4f}, valid_rmse {:.4f}, valid_mae {:.4f}, test_loss {:.4f}, test_rmse {:.4f}, test_mae {:.4f}"
              .format(df['train_loss'], df['valid_loss'], df['valid_metric_rmse'], df['valid_metric_mae'],
                                        df['test_loss'], df['test_metric_rmse'], df['test_metric_mae']))
        df.to_csv(config["checkpoint_path"] + "metric_mins.csv")

        # for col_name in myModel.df_metrics.keys():
        #     metric_mins[col_name] = myModel.df_metrics[col_name].min()
        # df = pd.DataFrame(metric_mins, index=[0])


def gaussian(size=(1, 1), **kwargs):
    if kwargs["params"] is None:
        return torch.normal(0, 1, size=size)
    else:
        noise = torch.normal(0, 1, size=size)
        if "scale" in kwargs["params"].keys():
            noise *= kwargs["params"]["scale"]
        if "loc" in kwargs["params"].keys():
            noise += kwargs["params"]["loc"]
        return noise


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true-y_pred)**2.0))


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true-y_pred))
