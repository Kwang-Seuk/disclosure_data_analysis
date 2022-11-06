import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def descriptive_statistics(df: DataFrame) -> DataFrame:
    df_descript = df.describe()
    df_descript.loc["skew"] = df.skew()
    df_descript.loc["kurt"] = df.kurt()
    return df_descript


def descriptive_statistics_groupby(df: DataFrame, group_var: str) -> DataFrame:

    df_grp_descript = (
        df.groupby(group_var)
        .agg(["mean", "std", "min", "max", "skew"])
        .unstack()
    )
    return df_grp_descript


def mean_std_boxplots(
    df: DataFrame, rows: int, cols: int, groupby: str
) -> None:

    fig, ax = plt.subplots(rows, cols, sharey=False)
    plt.suptitle("")
    df.boxplot(by=groupby, ax=ax)
