import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection
from xgboost import XGBRegressor, plot_tree

## Load data
data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/"
data_file = "rawdata_analysis_employment.csv"
df = pd.read_csv(data_dir + data_file, index_col=0)

## Data preparation for modelling

# Data preparation for model development
df_model = df.drop(["Yr_disclosure", "Regions", "Codes"], axis=1)
y = df_model["Employment_rates"]
X = df_model.drop(["Employment_rates"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=268, shuffle=False
)

# Forward input feature selection test
xgbm_sfs = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    tree_method="gpu_hist",
)

sfs_res = sfs(
    xgbm_sfs,
    k_features=88,
    forward=True,
    floating=False,
    verbose=2,
    scoring="neg_mean_squared_error",
    cv=5,
)

sfs_res = sfs_res.fit(X_train, y_train)

xgbm_sfs.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric=["rmse"],
    verbose=100,
)

fig = plot_sequential_feature_selection(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential forward Selection')
plt.rcParams["figure.figsize"] = (15,15)
plt.grid()
plt.show()



# plt.rcParams["figure.figsize"] = (50,50)
# plot_tree(xgbm, num_trees = xgbm.get_booster().best_iteration)
# plt.show()

tr_pred = xgbm.predict(X_train)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_train, tr_pred)

tst_pred = xgbm.predict(X_test)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_test, tst_pred)

# xgbm.feature_importances_
# feature_names=np.array(X_train.columns)
# sorted_idx = xgbm.feature_importances_.argsort()
# plt.figure(figsize=(15,50))
# plt.barh(feature_names[sorted_idx], xgbm.feature_importances_[sorted_idx])
# plt.xlabel("XGBoost Feature Importance")

perm_importance = permutation_importance(xgbm, X_train, y_train)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(15, 50))
plt.barh(
    feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
)
plt.xlabel("Permutation Importance")
