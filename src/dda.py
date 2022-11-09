import pandas as pd
import numpy as np
import seaborn as sns
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

def descriptive_statistics(df: DataFrame) -> DataFrame:
    df_descript = df.describe()
    df_descript["skew"] = df.skew()
    df_descript["kurt"] = df.kurt()
    return df_descript

def descriptive_statistics_groupby(
    df: DataFrame,
    group_var: str,
    remove_var1: str,
    remove_var2: str,
) -> DataFrame:

    df_tmp = df.drop([remove_var1, remove_var2], axis = 1)
    df_grp_descript = (
        df_tmp.groupby(group_var)
        .agg(["mean", "std", "min", "max", "skew"])
        .unstack()
    )
    return df_grp_descript

def correlation_matrix_figure(df: DataFrame, annot: str) -> None:
    correlations = df.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, vmax=1.0, center=0,
                fmt='.2f', cmap="YlGnBu", square=True,
                linewidths=.5, annot=annot) #char_kws = {'shrink': .70})
    plt.show


def mean_std_boxplots(
    df: DataFrame, rows: int, cols: int, groupby: str
) -> None:

    fig, ax = plt.subplots(figsize=(15, 50), sharey=False)
    plt.suptitle("")
    df.boxplot(by=groupby, ax=ax)

