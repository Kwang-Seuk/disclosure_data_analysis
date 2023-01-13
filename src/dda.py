# Loading modules
import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from matplotlib import pyplot as plt
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
    x_train: DataFrame, x_test: DataFrame, y_train: Series, sel_trial: int
) -> DataFrame:

    # Preliminary XGBoost model creation
    xgbm_pre = XGBRegressor(
        objective="reg:squarederror", tree_method="gpu_hist"
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
        tree_method="gpu_hist",
        n_jobs=-1,
    )
    best_score = cross_val_score(
        xgb_hpo,
        hpspace["x_train"],
        hpspace["y_train"],
        scoring="neg_mean_squared_error",
        cv=10,
    ).mean()
    loss = 1 - best_score
    return loss


def develop_production_model(data_dict: dict, best: dict):
    x_train = data_dict["x_train"]
    x_test = data_dict["x_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    best["max_depth"] = int(best["max_depth"])

    xgb_model_production = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        n_jobs=1,
        **best,
        tree_method="gpu_hist",
    )

    xgb_model_production.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric=["rmse"],
        verbose=100,
        early_stopping_rounds=400,
    )

    return xgb_model_production


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
    fig_save: bool = True,
):

    col_list = list(df_itp.columns)

    fig = plt.figure()
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


# Random simulation functions
def minmax_table(data_dict: dict, rdn_num: int):

    x_train = data_dict["x_train"]

    minmax_df = pd.DataFrame(x_train.nunique(), columns=["nunique"])
    minmax_df["max"] = x_train.max()
    minmax_df["min"] = x_train.min()
    minmax_df["dtypes"] = x_train.dtypes

    df_rdn = pd.DataFrame()

    for col_name in minmax_df.index:
        if minmax_df["dtypes"][col_name] == "float64":
            df_rdn[col_name] = (
                np.random.rand(rdn_num)
                * (minmax_df["max"][col_name] - minmax_df["min"][col_name])
                + minmax_df["min"][col_name]
            )
        else:
            df_rdn[col_name] = np.random.randint(
                minmax_df["min"][col_name],
                minmax_df["max"][col_name] + 1,
                size=rdn_num,
            )

    return minmax_df, df_rdn


def rdn_simul_data_create(
    minmax_df: DataFrame,
    df_rdn: DataFrame,
    linnum: int,
    save_res: bool = True,
):

    df_tmp = pd.DataFrame()
    df_rdn_dict = {}  # <-- newly added / not working well

    for idx, col_name in enumerate(minmax_df.index):

        # If the number of unique values of a variable is > linnum,
        # make values equals to linnum, otherwise number of values = nunique
        if minmax_df["nunique"][col_name] > linnum:
            temp_linspace = np.linspace(
                minmax_df["min"][col_name], minmax_df["max"][col_name], linnum
            )
        else:
            temp_linspace = np.linspace(
                minmax_df["min"][col_name],
                minmax_df["max"][col_name],
                minmax_df["nunique"][col_name],
            )

        temp_name = "df_rdn_" + col_name
        globals()[temp_name] = pd.DataFrame()

        for idx, value in enumerate(temp_linspace):
            df_temp = df_rdn.copy()
            df_temp[col_name] = value

            if idx == 0:
                temp_name = df_temp.copy()
            else:
                temp_name = temp_name.append(df_temp)
                df_dict = {
                    col_name: temp_name
                }  # <-- newly added / not                working well

        # if col_name == 'Male':
        #  temp_name['Female'] =  np.where( temp_name['Male'] ==0, 1, 0)
        # if col_name == 'Female':
        #  temp_name['Male'] =  np.where( temp_name['Female'] ==0, 1, 0)

        # if idx == 0:
        #    df_tmp = temp_name.copy()
        # else:
        #    df_tmp = df_tmp.append(temp_name)

        # Make CSV files
        print(
            "df_rdn_"
            + col_name
            + " done. "
            + str(temp_name.shape[0])
            + " rows"
        )
        # print(temp_name.head())
        if save_res is True:
            cwd = os.getcwd()
            temp_name.to_csv(cwd + "/df_rdn_" + col_name + ".csv")

    print("All input features are processed")
    return df_tmp, df_rdn_dict
