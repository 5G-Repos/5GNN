
import torch
import shapely
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd


GreenToRed = mpl.colors.LinearSegmentedColormap.from_list('GreenToRed',
                                                          [(0,    '#00ff00'),
                                                           (1,    '#ff2a00')], N=256)

RedToGreen = mpl.colors.LinearSegmentedColormap.from_list('RedToGreen',
                                                          [(0,    '#ff2a00'),
                                                           (1,    '#00ff00'),], N=256)

RedToGreenToBlue = mpl.colors.LinearSegmentedColormap.from_list('RedToGreen',
                                                                [(0,    '#F24A29'),
                                                                 (0.5,  '#1DA840'),
                                                                 (1,    '#000080')], N=256)


def plot_loss_curve(myModel, config):
    """
    The line chart for the training loss and test loss over all epochs.
    """

    if len(myModel.df_metrics) == 1:
        print("Skip the plot generation since only one epoch.")
        return
    if np.all(np.isnan(myModel.df_metrics.values.astype(float))):
        print("Skip the plot generation since no valid values.")
        return
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        for col_name in myModel.df_metrics.keys():
            if "loss" in col_name:
                myModel.df_metrics[col_name].dropna().plot(ax=ax1, alpha=0.7, grid=True)
        ax1.set_yticks(np.linspace(start=0, stop=0.4, num=9))
        ax1.set_ylim(-0.01, 0.5)
        ax1.set_title("Model Loss")
        ax1.legend()
        fig.savefig(config["checkpoint_path"] + "Summary_loss.png", dpi=100, transparent=True, bbox_inches="tight")


def plot_metric_curve(myModel, config):
    """
    The line chart for different evaluation metrics over all epochs.
    """

    if len(myModel.df_metrics) == 1:
        print("Skip the plot generation since only one epoch.")
        return
    if np.all(np.isnan(myModel.df_metrics.values.astype(float))):
        print("Skip the plot generation since no valid values.")
        return
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        for col_name in myModel.df_metrics.keys():
            if "metric" in col_name:
                myModel.df_metrics[col_name].dropna().plot(ax=ax1, alpha=0.7, grid=True)
        ax1.set_title("Model Metrics")
        ax1.legend()
        fig.savefig(config["checkpoint_path"] + "Summary_metrics.png", dpi=100, transparent=True, bbox_inches="tight")


def draw_data_points(epoch, total_y, train_c, valid_c, test_c, config):
    """
    The scatter plot of data points. This function will be called every certain epoches during the training.
    :param:
        total_y: tensor of all the labels
        train_c: tensor of coord of training dataset
        test_c: tensor of coord of test dataset
    """

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, label in enumerate(range(total_y.shape[1])):
        # get the color from model's output
        norm_out = (total_y[:, i] - min(total_y[:, i])) / (max(total_y[:, i]) - min(total_y[:, i]))
        # colors = cm.rainbow(norm_out)

        # plot obs points
        #obs_lat, obs_lon = train_c[:, 0], train_c[:, 1]
        #ax.scatter(obs_lat, obs_lon, c=norm_out[:len(train_c)], cmap=RedToGreen, marker='o', s=config["point_size"], vmin=0.5, vmax=1)
        ## obs_color = colors[:len(train_c)]
        ## ax.scatter(obs_lat, obs_lon, color=np.array(obs_color), marker='o', s=config["point_size"])

        # plot pred points
        #test_lat, test_lon = test_c[:, 0], test_c[:, 1]
        #ax.scatter(test_lat, test_lon, c=norm_out[len(train_c):], cmap=RedToGreen, marker='*', s=config["point_size"], vmin=0.5, vmax=1)
        ## test_color = colors[len(train_c):]
        ## ax.scatter(test_lat, test_lon, color=np.array(test_color), marker='*', s=config["point_size"])

        # plot total points
        total_c = np.concatenate((train_c, valid_c, test_c), axis=0)
        total_lat, total_lon = total_c[:, 0], total_c[:, 1]
        ax.scatter(total_lat, total_lon, c=norm_out, cmap=RedToGreen, marker='o', s=config["point_size"], vmin=0.3, vmax=0.9)

        # generate plot
        ax.set_xlabel(r'$c^{(1)}$', fontsize=14)
        ax.set_ylabel(r'$c^{(2)}$', fontsize=14)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_title("Model - Epoch " + str(epoch) + " - Variable " + str(label))
        plt.savefig(config["checkpoint_path"] + str(label) + "_iter_" + str(epoch) + ".png", dpi=100, transparent=True, bbox_inches="tight")


def draw_imputation_map(model, num_grid, total_C, train_C, config):
    """
    model: trained model
    min_x, min_y, max_x, max_y: boundary of area to imputation
    total_C: observed coordinates
    """

    # process coordinates to geo_dataframe
    if type(total_C) == torch.Tensor:
        total_C = total_C.numpy()
    total_C = config["scale_c"].transform(total_C)
    total_x, total_y = np.split(total_C, 2, axis=1)
    total_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(total_x.reshape(-1), total_y.reshape(-1)))
    xmin, ymin, xmax, ymax = total_gdf.total_bounds

    if type(train_C) == torch.Tensor:
        train_C = train_C.numpy()
    train_x, train_y = np.split(train_C, 2, axis=1)
    train_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(train_x.reshape(-1), train_y.reshape(-1)))

    # n x n grids
    n_cells = num_grid
    cell_size = (xmax - xmin) / n_cells

    # create grid cells
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])

    # get all centroids of each grid cell
    centroids = [i.centroid for i in cell.geometry]

    # Get X and Y coordinates of centroids of cells
    cell_x = [i.x for i in centroids]
    cell_y = [i.y for i in centroids]

    # Create list of XY coordinate pairs
    cell_cent_coords = np.array([list(xy) for xy in zip(cell_x, cell_y)])

    degree = config["scale_x"].transform(np.array([[0, 0, 330]]))[0, -1]
    # feats = np.zeros((cell_cent_coords.shape[0], 1))
    feats = np.ones((cell_cent_coords.shape[0], 1))*degree
    cell_cent_feats = np.concatenate((cell_cent_coords, feats), axis=1)

    # cell_cent_coords = config["scale_c"].transform(cell_cent_coords)
    # cell_cent_feats = config["scale_x"].transform(cell_cent_feats)

    pred_cell = model.predict(cell_cent_coords, cell_cent_feats)

    cell['value'] = pred_cell

    # green to red cmap
    GreenToRed = mpl.colors.LinearSegmentedColormap.from_list('GreenToRed', [(0, '#ff2a00'),
                                                                             (1, '#00ff00')], N=256)

    # plot the grid
    ax = cell.plot(column='value', figsize=(14, 10), cmap=GreenToRed, vmax=cell['value'].max(), edgecolor="grey")  # , legend=True
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.autoscale(False)
    train_gdf.plot(ax=ax, marker='o', color='dimgray', markersize=2)

    plt.savefig(config["checkpoint_path"] + "map" + ".png", dpi=100, transparent=True, bbox_inches="tight")

