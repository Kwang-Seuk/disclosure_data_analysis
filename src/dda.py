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
from scipy.stats import kurtosis

def descriptive_statistics_groupby(
    df: DataFrame,
    group_var: str,
) -> DataFrame:

    df_grp_descript = (
        df.groupby(group_var)
        .agg(["mean", "std", "min", "max", "skew", kurtosis])
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
    # df_vif = df.drop(["Yr_disclosure", "Regions", "Codes"], axis=1)
    vif = pd.DataFrame()
    vif["features"] = df.columns
    vif["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif


def min_max_linspace_for_mip(df: DataFrame, interval: int) -> DataFrame:

    minmax_df = pd.DataFrame(df.nunique(), columns=["nunique"])
    minmax_df["min"] = df.min()
    minmax_df["max"] = df.max()
    minmax_df["dtypes"] = df.dtypes

    df_interpolated = pd.DataFrame()
    for idx, col_name in enumerate(minmax_df.index):
        temp_linspace = np.linspace(
            minmax_df["min"][col_name],
            minmax_df["max"][col_name],
            interval
        )
        df_interpolated[col_name] = temp_linspace    
    return df_interpolated

#def min_max_linspace_for_mip(df: DataFrame, interval: int) -> DataFrame:
#
#    minmax_df = pd.DataFrame(df.nunique(), columns=["nunique"])
#    minmax_df["min"] = df.min()
#    minmax_df["max"] = df.max()
#    minmax_df["dtypes"] = df.dtypes
#
#    df_interpolated = pd.DataFrame()
#    for idx, col_name in enumerate(minmax_df.index):
#
#        # If the number of unique values of a variable is > linnum,
#        # make values equals to linnum, otherwise number of values = nunique
#        if minmax_df["nunique"][col_name] > interval:
#            temp_linspace = np.linspace(
#                minmax_df["min"][col_name],
#                minmax_df["max"][col_name],
#                interval,
#            )
#        else:
#            temp_linspace = np.linspace(
#                minmax_df["min"][col_name],
#                minmax_df["max"][col_name],
#                minmax_df["nunique"][col_name],
#            )
#        df_interpolated[col_name] = temp_linspace
#    return df_interpolated

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


def run_mip_analysis_with_df_mip(df_mip: DataFrame, model: str, data_dir: str) -> None:

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
        X_mip = X_mip.merge(X_itp_series.to_frame(), left_index = True, right_index = True)

        # prediction with the created mip data and store the result to df_mip_res
        mip_col_name = X_mip.columns[-1]
        mip_pred = model.predict(X_mip)
        df_mip_res[mip_col_name] = mip_pred.tolist()

        # save the X_mip as a csv file with its itp column name
        X_mip["res"] = mip_pred.tolist()
        X_mip.to_csv(data_dir + "x_mip_" + mip_col_name + ".csv") 
        #vars()["X_mip_" + mip_col_name] = X_mip

    return X_itp_for_plot, df_mip_res

def plot_mip_analysis_results(
    df_mip_input: DataFrame,
    df_mip_res: DataFrame
):

    no_input_feat = len(df_mip_input.columns)
    params = {
        "font.size": 13.0,
        "axes.titlesize": 'medium',
        "figure.figsize": (15, 10),
        "axes.grid": True,
        #"figure.dpi": 75
    }
    plt.rcParams.update(params)
    
    col_list = list(df_mip_input.columns)
    subplot_titles = ["{}".format(col) for col in col_list]

    fig = plt.figure()
    for i in range(no_input_feat):
        ax = fig.add_subplot(5, 5, 1 + i)
        ax.plot(df_mip_input.iloc[:, i], df_mip_res.iloc[:, i], 'r-')
        ax.set_title(subplot_titles[i])
    fig.tight_layout()


## The below functions are not currently available (under revision)

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
