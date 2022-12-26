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
    descriptive_statistics_groupby as dstatg,
    correlation_matrix_figure,
    mean_std_boxplots,
)


## Load data
data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/"
data_file = "rawdata_analysis_employment.csv"
df = pd.read_csv(data_dir + data_file, index_col=0)


## Descriptive and simple statistics


# Descriptive statistics: mean, std, max, min, and skewness per group_var
dstatg_df = dstatg(df, "disclosure_year")
dstatg_df.to_csv(data_dir + "g_Yr_dstat.csv")

## Data statistical illustrations

# Box-whisker plots for the data per group_var
mean_std_boxplots(df, 11, 9, "Yr_disclosure")

# Histrogram illustration for all data
df_hist = df.drop(["Regions", "Codes"], axis=1)

plt.rcParams["figure.dpi"] = 300
df_hist['Number_of_capstone_students'].hist(by=df_hist['Yr_disclosure'], figsize = (10, 3), sharey = True, bins=20, layout=(1,3), rwidth=0.9, xrot = 360)
df_hist['Capstone_funds'].hist(by=df_hist['Yr_disclosure'], figsize = (10, 3), sharey = True, bins=20, layout=(1,3), rwidth=0.9, xrot = 360)
df_hist['TPTC_per_1K'].hist(by=df_hist['Yr_disclosure'], figsize = (10, 3), sharey = True, bins=20, layout=(1,3), rwidth=0.9, xrot = 360)
df_hist['TPTC'].hist(by=df_hist['Yr_disclosure'], figsize = (10, 3), sharey = True, bins=20, layout=(1,3), rwidth=0.9, xrot = 360)
# Corrlation matrix plot for all data
correlation_matrix_figure(df, False)
