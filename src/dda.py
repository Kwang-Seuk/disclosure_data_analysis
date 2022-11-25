import pandas as pd
import numpy as np
import seaborn as sns
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from xgboost import XGBRegressor, plot_tree
from sklearn.model_selection import cross_val_score

def descriptive_statistics(df: DataFrame) -> DataFrame:
    df_descript = df.describe()
    col_name = list(df_descript.columns)
    for i in col_name:
        df_descript.index["skew"][col_name] = df.skew()
        df_descript.index["kurt"][col_name] = df.kurt()
    return df_descript


def descriptive_statistics_groupby(
    df: DataFrame,
    group_var: str,
    remove_var1: str,
    remove_var2: str,
) -> DataFrame:

    df_tmp = df.drop([remove_var1, remove_var2], axis=1)
    df_grp_descript = (
        df_tmp.groupby(group_var)
        .agg(["mean", "std", "min", "max", "skew"])
        .unstack()
    )
    return df_grp_descript


def correlation_matrix_figure(df: DataFrame, annot: str) -> None:
    correlations = df.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(
        correlations,
        vmax=1.0,
        center=0,
        fmt=".2f",
        cmap="YlGnBu",
        square=True,
        linewidths=0.5,
        annot=annot,
    )  # char_kws = {'shrink': .70})
    plt.show


def mean_std_boxplots(
    df: DataFrame, rows: int, cols: int, groupby: str
) -> None:

    fig, ax = plt.subplots(figsize=(15, 50), sharey=False)
    plt.suptitle("")
    df.boxplot(by=groupby, ax=ax)


def compute_vif_for_X(df: DataFrame) -> DataFrame:
    #df_vif = df.drop(["Yr_disclosure", "Regions", "Codes"], axis=1)
    vif = pd.DataFrame()
    vif["features"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i)
        for i in range(df.shape[1])
    ]
    return vif

def create_df_filled_with_means(df: DataFrame) -> DataFrame:
    df_mean = pd.DataFrame(df.mean(), columns = ["mean"])
    
    df_mip = pd.DataFrame(columns = list(df.columns), index = range(len(df)))
    for col_name in df_mean.index:
        df_mip[col_name] = df_mean["mean"][col_name]
    return df_mip


def min_max_linspace_for_mip(df: DataFrame, interval: int) -> DataFrame:
    
    minmax_df = pd.DataFrame(df.nunique(), columns=["nunique"])
    minmax_df["min"] = df.min()
    minmax_df["max"] = df.max()
    minmax_df["dtypes"] = df.dtypes

    df_interpolated = pd.DataFrame()
    for idx, col_name in enumerate(minmax_df.index):

        # If the number of unique values of a variable is > linnum,
        # make values equals to linnum, otherwise number of values = nunique
        if minmax_df["nunique"][col_name] > interval:
            temp_linspace = np.linspace(
                minmax_df["min"][col_name], minmax_df["max"][col_name], interval
            )
        else:
            temp_linspace = np.linspace(
                minmax_df["min"][col_name],
                minmax_df["max"][col_name],
                minmax_df["nunique"][col_name],
            )
        df_interpolated[col_name] = temp_linspace
    return df_interpolated

def creating_mip_data_per_input_feature(df: DataFrame) -> DataFrame:
    df_mip = create_df_filled_with_means(df)
    for col_name in df.columns:
        df_mip_itp = df_mip.copy()
        df_mip_itp.loc[:, col_name] = df.loc[:, col_name]
        vars()["df_mip_itp_" + col_name] = df_mip_itp



def hyper_parameters_objective(hpspace: dict, X_train: DataFrame, y_train: DataFrame):
    xgb_hpo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
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
        xgb_hpo, X_train, y_train, scoring="neg_mean_squared_error", cv=10
    ).mean()
    loss = 1 - best_score
    return loss


def minmax_table(df: DataFrame, linnum: int, rdn_num: int, out_dir: str):

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

    minmax_df.to_csv(out_dir + "minmax_df.csv")
    df_rdn.to_csv(out_dir + "df_rdn.csv")

    return minmax_df, df_rdn


def rdn_simul_data_create(
    df: DataFrame,
    minmax_df: DataFrame,
    df_rdn: DataFrame,
    linnum: int,
    rdn_num: int,
    print_option: str,
    out_dir: str,
):

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
            df = temp_name.copy()
        else:
            df = df.append(temp_name)

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
