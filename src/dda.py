## Loading modules -----------------------------------------------------------
# The belows are the modules used in this function code.
## ---------------------------------------------------------------------------


# Fundamental data manipulation
import os
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.stats import kurtosis

# Illustration
from matplotlib import pyplot as plt
import seaborn as sns

# ML data preprocessing modules
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from BorutaShap import BorutaShap
from sklearn.base import clone

# XGBoost model development
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from xgboost import XGBRegressor, plot_tree
from sklearn.model_selection import cross_val_score


## Model development functions------------------------------------------
# These functions are for development of XGBoost model with data set.
# Two seperate functions for input feature selection is prepared: i.e.
# forward_seq_feat_selec() and feat_selec_with_borutashap(). Either can be
# used in your model development.
## ---------------------------------------------------------------------


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
    X_train: DataFrame, X_test: DataFrame, y_train: Series
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
        X=X_train,
        y=y_train,
        n_trials=100,
        sample=False,
        train_or_test="train",
        normalize=True,
        verbose=False,
    )

    # Create new input features sets for training and testing
    features_to_remove = Feature_Selector.features_to_remove
    X_train_boruta_shap = X_train.drop(columns=features_to_remove)
    X_test_boruta_shap = X_test.drop(columns=features_to_remove)

    print(
        "The number of selected input features is ",
        len(X_train_boruta_shap.columns),
        "including...",
        X_train_boruta_shap.columns,
    )

    return X_train_boruta_shap, X_test_boruta_shap


def compute_vif_for_X(df: DataFrame) -> DataFrame:
    # df_vif = df.drop(["Yr_disclosure", "Regions", "Codes"], axis=1)
    vif = pd.DataFrame()
    vif["features"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif


def develop_your_production_model(
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    best: dict,
):

    for key, value in best.items():
        best["max_depth"] = int(best["max_depth"])
        # best['n_estimators']=int(best['n_estimators'])

    xgb_model_production = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        n_jobs=-1,
        **best,
        tree_method="gpu_hist"
    )

    xgb_model_production.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["rmse"],
        verbose=100,
        early_stopping_rounds=400,
    )

    return xgb_model_production


## Training results illustration functions ------------------------------------
# The followings enable you to investigate the trained production model
# structure and its predictability against training and testing data sets,
# by depicting the model output results
## ----------------------------------------------------------------------------


def best_tree_illustration(
    model, fig_size: tuple, fig_dpi: int, fig_save: bool
):

    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.dpi"] = fig_dpi
    plot_tree(
        model,
        num_trees=model.get_booster().best_iteration,
    )
    if fig_save == True:
        cwd = os.getcwd()
        plt.savefig(cwd + "/fig_test_tree.jpg", dpi="figure")


def predict_train_test(
    model,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    fig_save: bool,
):

    tr_pred = model.predict(X_train)
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.scatter(y_train, tr_pred)

    tst_pred = model.predict(X_test)
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.scatter(y_test, tst_pred)

    if fig_save == True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "fig_train_test_prediction_results.png", dpi="figure"
        )


def feat_importance_general(model, X_train: DataFrame):
    model.feature_importances_
    feature_names = np.array(X_train.columns)
    sorted_idx = model.feature_importances_.argsort()
    plt.figure(figsize=(15, 10))
    plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.xlabel("XGBoost Feature Importance")

    if fig_save == True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "fig_feature_importance.png", dpi="figure"
        )


def feat_importance_permut(model, X_train: DataFrame, y_train: Series):
    perm_importance = permutation_importance(model, X_train, y_train)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(15, 10))
    plt.barh(
        feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
    )
    plt.xlabel("Permutation Importance")

    if fig_save == True:
        cwd = os.getcwd()
        plt.savefig(
            cwd + "fig_feature_importance_permuted.png",
            dpi="figure",
        )


## Most-influencing parameter -------------------------------------------------
# The functions generate min-max ranged linear values for every input feature
# selected by the modelling process. A dataframe consisting of One linearly
# varying feature and average values for the remaining features is generated
# with the following functions. Then you can campare  responses of the trained
# model against the given gradual variation of each input feature.
## ----------------------------------------------------------------------------


def min_max_linspace_for_mip(df: DataFrame, interval: int) -> DataFrame:

    minmax_df = pd.DataFrame(df.nunique(), columns=["nunique"])
    minmax_df["min"] = df.min()
    minmax_df["max"] = df.max()
    minmax_df["dtypes"] = df.dtypes

    df_interpolated = pd.DataFrame()
    for idx, col_name in enumerate(minmax_df.index):
        temp_linspace = np.linspace(
            minmax_df["min"][col_name], minmax_df["max"][col_name], interval
        )
        df_interpolated[col_name] = temp_linspace
    return df_interpolated


def create_df_mip_with_means_and_itp_data(
    df: DataFrame, df_interpolated: DataFrame
) -> DataFrame:

    df_mean_tmp = pd.DataFrame(df.mean(), columns=["mean"])
    df_means = pd.DataFrame(
        columns=list(df.columns), index=range(len(df_interpolated))
    )
    for col_name in df_mean_tmp.index:
        df_means[col_name] = df_mean_tmp["mean"][col_name]
    df_means = df_means.add_suffix("_means")
    df_mip = df_interpolated.join(df_means)
    return df_mip


def run_mip_analysis_with_df_mip(
    df_mip: DataFrame, model: str, data_dir: str
) -> None:

    X_grp_no = int(len(df_mip.columns) / 2)
    X_itp = df_mip.iloc[:, 0:X_grp_no]
    X_means = df_mip.iloc[:, X_grp_no:]

    # store data for further plot creation
    df_mip_res = pd.DataFrame()
    X_itp_for_plot = X_itp.copy()

    for i in range(X_grp_no):
        # make an mip data set for prediction
        X_itp_tmp = X_itp.copy()
        X_means_tmp = X_means.copy()
        X_mip = X_means_tmp.drop(X_means_tmp.columns[i], axis=1)
        X_itp_series = X_itp_tmp.iloc[:, i]
        X_mip = X_mip.merge(
            X_itp_series.to_frame(), left_index=True, right_index=True
        )

        # prediction with the created mip data and store the result to df_mip_res
        mip_col_name = X_mip.columns[-1]
        mip_pred = model.predict(X_mip)
        df_mip_res[mip_col_name] = mip_pred.tolist()

        # save the X_mip as a csv file with its itp column name
        X_mip["res"] = mip_pred.tolist()
        X_mip.to_csv(data_dir + "x_mip_" + mip_col_name + ".csv")
        # vars()["X_mip_" + mip_col_name] = X_mip

    return X_itp_for_plot, df_mip_res


def plot_mip_analysis_results(df_mip_input: DataFrame, df_mip_res: DataFrame):

    no_input_feat = len(df_mip_input.columns)
    params = {
        "font.size": 13.0,
        "axes.titlesize": "medium",
        "figure.figsize": (15, 10),
        "axes.grid": True,
        # "figure.dpi": 75
    }
    plt.rcParams.update(params)

    col_list = list(df_mip_input.columns)
    subplot_titles = ["{}".format(col) for col in col_list]

    fig = plt.figure()
    for i in range(no_input_feat):
        ax = fig.add_subplot(5, 5, 1 + i)
        ax.plot(df_mip_input.iloc[:, i], df_mip_res.iloc[:, i], "r-")
        ax.set_title(subplot_titles[i])
    fig.tight_layout()


## Randomized simulation functions--------------------------------------
# These functions are designed (1) to create min-max interpolated data
# for every input feature, and (2) to generate randomized data blocks
# consisted of the interpolated data and randomly distributed data.
##----------------------------------------------------------------------


def minmax_table(df: DataFrame, rdn_num: int):

    minmax_df = pd.DataFrame(df.nunique(), columns=["nunique"])
    minmax_df["max"] = df.max()
    minmax_df["min"] = df.min()
    minmax_df["dtypes"] = df.dtypes

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
    rdn_num: int,
    print_option: str,
    out_dir: str,
):

    df_tmp = pd.DataFrame()

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

        # if col_name == 'Male':
        #  temp_name['Female'] =  np.where( temp_name['Male'] ==0, 1, 0)
        # if col_name == 'Female':
        #  temp_name['Male'] =  np.where( temp_name['Female'] ==0, 1, 0)

        if idx == 0:
            df_tmp = temp_name.copy()
        else:
            df_tmp = df_tmp.append(temp_name)

        # Make CSV files
        print(
            "df_rdn_"
            + col_name
            + " done. "
            + str(temp_name.shape[0])
            + " rows"
        )
        # print(temp_name.head())
        if print_option == True:
            # temp_name.to_excel(out_data_dir + "df_rdn_" + col_name + ".xlsx")
            temp_name.to_csv(out_dir + "df_rdn_" + col_name + ".csv")
