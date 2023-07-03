# Loading modules
import os
import pandas as pd
import numpy as np
import math
from typing import Union
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from BorutaShap import BorutaShap
from xgboost import XGBRegressor, plot_tree


# Model development functions
def load_your_data(
    df: DataFrame, train_size: int, target_var: str
) -> DataFrame:
    df_model = df.copy()
    y = df_model[target_var]
    X = df_model.drop([target_var], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=False
    )

    return X_train, X_test, y_train, y_test


def feat_selec_with_borutashap(
    x_train: DataFrame,
    x_test: DataFrame,
    y_train: Series,
    sel_trial: int,
    gpu_flag: bool,
) -> DataFrame:
    if gpu_flag is True:
        xgbm_pre = XGBRegressor(
            objective="reg:squarederror", tree_method="gpu_hist"
        )
    else:
        xgbm_pre = XGBRegressor(
            objective="reg:squarederror", tree_method="approx"
        )

    # Making a feature selection frame with BorataShap module
    Feature_Selector = BorutaShap(
        model=xgbm_pre,
        importance_measure="shap",
        classification=False,
        percentile=100,
        pvalue=0.05,
    )

    # Fitting and selecting input features
    Feature_Selector.fit(
        X=x_train,
        y=y_train,
        n_trials=sel_trial,
        sample=False,
        train_or_test="train",
        normalize=True,
        verbose=False,
    )

    # Create new input features sets for training and testing
    features_to_remove = Feature_Selector.features_to_remove
    x_train_sel = x_train.drop(columns=features_to_remove)
    x_test_sel = x_test.drop(columns=features_to_remove)

    # Sort input features by column titles
    x_train_sel = x_train_sel.reindex(sorted(x_train_sel.columns), axis=1)
    x_test_sel = x_test_sel.reindex(sorted(x_test_sel.columns), axis=1)

    print(
        "The number of selected input features is ",
        len(x_train_sel.columns),
        "including...",
        str(x_train_sel.columns),
    )

    return x_train_sel, x_test_sel


def hyper_parameters_objective(hpspace: dict):
    xgb_hpo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        max_depth=int(hpspace["max_depth"]),
        gamma=hpspace["gamma"],
        reg_alpha=hpspace["reg_alpha"],
        reg_lambda=hpspace["reg_lambda"],
        eta=hpspace["eta"],
        min_child_weight=hpspace["min_child_weight"],
        subsample=hpspace["subsample"],
        colsample_bytree=hpspace["colsample_bytree"],
        scale_pos_weight=hpspace["scale_pos_weight"],
        n_jobs=1,
    )  # tree_method="gpu_hist",

    best_score = cross_val_score(
        xgb_hpo,
        hpspace["x_train"],
        hpspace["y_train"],
        scoring="neg_mean_squared_error",
        cv=10,
    ).mean()
    loss = 1 - best_score
    return loss


def develop_production_model(
    data_dict: dict, iterations: int, best: dict, gpu_flag: bool
):
    x_train = data_dict["x_train"]
    x_test = data_dict["x_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    best["max_depth"] = int(best["max_depth"])

    if gpu_flag is True:
        xgb_model_production = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=iterations,
            n_jobs=1,
            **best,
            tree_method="gpu_hist",
        )
    else:
        xgb_model_production = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=iterations,
            n_jobs=1,
            **best,
            tree_method="approx",
        )

    xgb_model_production.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric=["rmse"],
        verbose=100,
        early_stopping_rounds=15,
    )

    return xgb_model_production


def production_model_rmse_display(model):
    learning_res = model.evals_result()
    epochs = len(learning_res["validation_0"]["rmse"])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, learning_res["validation_0"]["rmse"], label="Train")
    ax.plot(x_axis, learning_res["validation_1"]["rmse"], label="Test")
    ax.legend()
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.title("XGBoost RMSE")
    plt.show()


# Training results illustration functions
def best_tree_illustration(model, fig_dpi: int, fig_save: bool = True):
    plt.rcParams["figure.dpi"] = fig_dpi
    plot_tree(
        model,
        num_trees=model.get_booster().best_iteration,
    )
    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(cwd + "/fig_test_tree.jpg", dpi=fig_dpi)


def predict_plot_train_test(
    model, data_dict: dict, fig_dpi: int, fig_save: bool = True
):
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train = data_dict["x_train"]
    x_test = data_dict["x_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    tr_pred = model.predict(x_train)
    plt.scatter(y_train, tr_pred)

    tst_pred = model.predict(x_test)
    plt.scatter(y_test, tst_pred)

    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_train_test_prediction_results.png", dpi=fig_dpi
        )


def feat_importance_general(
    model, data_dict: dict, fig_dpi: int, fig_save: bool = True
):
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train = data_dict["x_train"]
    feature_names = np.array(x_train.columns)
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.xlabel("XGBoost Feature Importance")

    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(cwd + "/fig_feature_importance.png", dpi=fig_dpi)


def feat_importance_permut(
    model, data_dict: dict, fig_dpi: int, fig_save: bool = True
):
    plt.rcParams["figure.dpi"] = fig_dpi

    x_train = data_dict["x_train"]
    y_train = data_dict["y_train"]

    perm_importance = permutation_importance(model, x_train, y_train)
    feature_names = np.array(x_train.columns)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(
        feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
    )
    plt.xlabel("Permutation Importance")

    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_feature_importance_permuted.png",
            dpi=fig_dpi,
        )


# Most-influencing parameter
def create_interpolation_for_mip(data_dict: dict, interval: int) -> DataFrame:
    x_train = data_dict["x_train"]

    minmax_df = pd.DataFrame(x_train.nunique(), columns=["nunique"])
    minmax_df["min"] = x_train.min()
    minmax_df["max"] = x_train.max()
    minmax_df["dtypes"] = x_train.dtypes

    df_interpolated = pd.DataFrame()
    for _, col_name in enumerate(minmax_df.index):
        temp_linspace = np.linspace(
            minmax_df["min"][col_name], minmax_df["max"][col_name], interval
        )
        df_interpolated[col_name] = temp_linspace

    return df_interpolated


def create_df_means_for_mip(
    data_dict: dict, df_interpolated: DataFrame
) -> DataFrame:
    x_train = data_dict["x_train"]

    df_mean_tmp = pd.DataFrame(x_train.mean(), columns=["mean"])
    df_means = pd.DataFrame(
        columns=list(x_train.columns), index=range(len(df_interpolated))
    )
    for col_name in df_mean_tmp.index:
        df_means[col_name] = df_mean_tmp["mean"][col_name]

    return df_means


def mip_analysis(
    df_means: DataFrame, df_itp: DataFrame, model: str, save_res: bool = True
):
    # store data for further plot creation
    df_mip_res = pd.DataFrame()
    df_means_copy = df_means.copy()

    for col_name in df_means.columns:
        df_means_copy[col_name] = df_itp[col_name]
        mip_pred = model.predict(df_means_copy)
        df_mip_res[col_name] = mip_pred.tolist()

    # save the mip results as a csv file
    if save_res is True:
        cwd = os.getcwd()
        df_mip_res.to_csv(cwd + "/mip_analysis_res" + ".csv")

    return df_mip_res


def plot_mip_analysis_results(
    df_itp: DataFrame,
    df_mip_res: DataFrame,
    fig_dpi: int,
    fig_size: tuple,
    fig_save: bool = True,
):
    col_list = list(df_itp.columns)

    fig = plt.figure(figsize=fig_size)
    for i, col_name in enumerate(col_list):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.plot(df_itp[col_name], df_mip_res[col_name], "r-")
        ax.set_title(col_name)
        fig.tight_layout()

    if fig_save is True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "/fig_mip_results.png",
            dpi=fig_dpi,
        )


# randomized simulation
def create_random_dataframes(df_interpolated: DataFrame, n: int) -> None:
    interval = df_interpolated.shape[0]

    df_rand = pd.DataFrame(
        {
            column: np.random.uniform(
                df_interpolated[column].min(), df_interpolated[column].max(), n
            )
            for column in df_interpolated.columns
        }
    )
    df_rand_tmp = pd.concat([df_rand] * interval, ignore_index=True)
    df_interval_tmp = pd.DataFrame(
        {
            column: np.repeat(df_interpolated[column].values, n)
            for column in df_interpolated.columns
        }
    )
    df_frames = [
        df_rand_tmp.copy().assign(**{column: df_interval_tmp[column]})
        for column in df_interpolated.columns
    ]
    for i, df in enumerate(df_frames):
        filename = f"df_randomized_{df_interpolated.columns[i]}.csv"
        df.to_csv(filename, index=False)


def random_simulation(
    input_csv: str, model: Union[XGBRegressor, str], save_res: bool = True
) -> None:
    df_rand = pd.read_csv(input_csv)

    if isinstance(model, str):
        loaded_model = XGBRegressor()
        loaded_model.load_model(model)
        model = loaded_model

    rand_pred = model.predict(df_rand)
    df_rand["y"] = rand_pred.tolist()
    if save_res is True:
        cwd = os.getcwd()
        base = os.path.basename(input_csv)
        filename = os.path.splitext(base)[0]
        df_rand.to_csv(
            cwd + "/random_simulation_res_" + filename + ".csv", index=False
        )


def draw_scatter_graphs_from_csv(
    csv_file: str, control_var: str, x_var: str, y_var: str
) -> None:
    df = pd.read_csv(csv_file)

    interval = df[control_var].nunique()

    # Calculate subplot grid dimensions
    grid_size = math.ceil(math.sqrt(interval))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.ravel()  # Flatten axis array

    # Determine dot size based on dataframe size
    dot_size = max(5, 5000 / len(df))

    for subplot_index, unique_val in enumerate(df[control_var].unique()):
        subset = df[df[control_var] == unique_val]
        scatter = axs[subplot_index].scatter(
            subset[x_var],
            subset[y_var],
            c=subset["y"],
            cmap="jet",
            s=dot_size,
        )
        fig.colorbar(scatter, ax=axs[subplot_index])
        axs[subplot_index].set_title(f"{control_var} = {unique_val}")

        if (subplot_index + grid_size) // grid_size >= grid_size:
            axs[subplot_index].set_xlabel(x_var)

        if subplot_index % grid_size == 0:
            axs[subplot_index].set_ylabel(y_var)

    # Remove unused subplots
    for i in range(interval, grid_size * grid_size):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def draw_contour_graphs_from_csv(
    csv_file: str, control_var: str, x_var: str, y_var: str
) -> None:
    df = pd.read_csv(csv_file)
    interval = df[control_var].nunique()

    # Calculate subplot grid dimensions
    grid_size = math.ceil(math.sqrt(interval))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.ravel()  # Flatten axis array

    for subplot_index, unique_val in enumerate(df[control_var].unique()):
        subset = df[df[control_var] == unique_val]
        contour = axs[subplot_index].tricontourf(
            subset[x_var], subset[y_var], subset["y"], cmap="jet"
        )
        fig.colorbar(contour, ax=axs[subplot_index])
        axs[subplot_index].set_title(f"{control_var} = {unique_val}")

        if (
            subplot_index // grid_size == grid_size - 1
        ):  # Only label x-axis on the bottom-most subplots
            axs[subplot_index].set_xlabel(x_var)

        if (
            subplot_index % grid_size == 0
        ):  # Only label y-axis on the left-most subplots
            axs[subplot_index].set_ylabel(y_var)

    # Remove unused subplots
    for i in range(interval, grid_size * grid_size):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
