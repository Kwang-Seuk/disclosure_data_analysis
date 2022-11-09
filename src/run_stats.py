import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection
from xgboost import XGBRegressor, plot_tree
from src.dda import (
    descriptive_statistics as dstat,
    descriptive_statistics_groupby as gdstat,
    correlation_matrix_figure,
    mean_std_boxplots,
)


## Load data
data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/"
data_file = "rawdata_analysis_employment.csv"
df = pd.read_csv(data_dir + data_file, index_col=0)


## Descriptive and simple statistics

# Descriptive statistics: mean, std, max, min, skewness, and kurtosis for all data
dstat_df = dstat(df)
dstat_df.to_csv(data_dir + "dstat.csv")

# Descriptive statistics: mean, std, max, min, and skewness per group_var
gdstat_df = gdstat(df, "Yr_disclosure", "Regions", "Codes")
gdstat_df.to_csv(data_dir + "g_Yr_dstat.csv")

## Data statistical illustrations

# Box-whisker plots for the data per group_var
mean_std_boxplots(df, 11, 9, "Yr_disclosure")

# Histrogram illustration for all data
df_hist = df.drop(["Regions", "Codes"], axis=1)
df_hist.hist(figsize = (50, 25), bins=30)

# Corrlation matrix plot for all data
correlation_matrix_figure(df, False)
