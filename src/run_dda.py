import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection
from hyperopt import hp
from xgboost import XGBRegressor, plot_tree
from BorutaShap import BorutaShap
from sklearn.base import clone

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

## Preliminary model training

xgbm_pre = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    gamma=1,
    min_child_weight=1,
    colsample_bytree=1,
    max_depth=5,
)

xgbm_pre.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    early_stopping_rounds=400,
)

plt.rcParams["figure.figsize"] = (50, 50)
plot_tree(xgbm_pre, num_trees=xgbm_pre.get_booster().best_iteration)
plt.show()

## Shapley index based feature importance check (BorutaShap)

model_BS = clone(xgbm_pre)
Feature_Selector = BorutaShap(model = model_BS,
                              importance_measure = 'shap',
                              classification = True,
                              percentile = 100,
                              pvalue = 0.05)

Feature_Selector.fit(X=X_train, y=y_train,
                     n_trials = 100,
                     sample = False,
                     train_or_test = 'train',
                     normalize = True,
                     verbose = False
                     )

Feature_Selector.plot(figsize=(30, 10), which_features = 'all')


## Forward input feature selection test (hereafter new XGB created)

# Input feature exploration by sequential_feature_selector
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
    scoring="neg_root_mean_squared_error",
    cv=5,
)

sfs_res = sfs_res.fit(X_train, y_train)

# Plot negative RMSE against the number of input features used in the test
fig = plot_sequential_feature_selection(sfs_res.get_metric_dict(), kind="std_dev")
plt.title("Sequential forward Selection")
plt.rcParams["figure.figsize"] = (30, 20)
plt.grid()
plt.show()

# Input feature exploration by sequential_feature_selector (up to best features)
xgbm_sfs1 = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    tree_method="gpu_hist",
)

sfs_res1 = sfs(
    xgbm_sfs1,
    k_features=6,
    forward=True,
    floating=False,
    verbose=2,
    scoring="neg_root_mean_squared_error",
    cv=5,
)

sfs_res1 = sfs_res1.fit(X_train, y_train)


# Get the best input feature list from the sfs result
feat_cols = list(sfs_res1.k_feature_idx_)
print(feat_cols)

# Make new train/test data with selected input features
X_tr_sel = X_train.iloc[:, feat_cols]
X_tst_sel = X_test.iloc[:, feat_cols]

# New XGBoost model with the selected input features
xgbm_sel_if = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    gamma=1,
    min_child_weight=1,
    colsample_bytree=1,
    max_depth=5,
)

xgbm_sel_if.fit(
    X_tr_sel,
    y_train,
    eval_set=[(X_tr_sel, y_train), (X_tst_sel, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    early_stopping_rounds=400,
)

plt.rcParams["figure.figsize"] = (50, 50)
plot_tree(xgbm_sel_if, num_trees=xgbm_sel_if.get_booster().best_iteration)
plt.show()

## Prediction results (train / test data)

tr_sel_pred = xgbm_sel_if.predict(X_tr_sel)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_train, tr_sel_pred)

tst_sel_pred = xgbm_sel_if.predict(X_tst_sel)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_test, tst_sel_pred)

## Feature importance tests

# Feature importance test for the selected input features
xgbm_sel_if.feature_importances_
feature_names = np.array(X_tr_sel.columns)
sorted_idx = xgbm_sel_if.feature_importances_.argsort()
plt.figure(figsize=(15, 10))
plt.barh(feature_names[sorted_idx], xgbm_sel_if.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Feature Importance")

# Feature importance (permutation) test for the selected input features
perm_importance = permutation_importance(xgbm_sel_if, X_tr_sel, y_train)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(15, 10))
plt.barh(
    feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
)
plt.xlabel("Permutation Importance")


## The code below is incomplete

## Hyper-parameters optimization

# Setting up hyper-parameter space
hpspace = {
    "max_depth": hp.quniform("max_depth", 3, 15, 1),
    "n_estimators": 180,
    "gamma": hp.uniform("gamma", 1, 9),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "eta": hp.uniform("eta", 0.01, 0.2),
    "min_child_weight": hp.quniform("min_child_weight", 0, 2, 0.1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "reg_alpha": hp.quniform("reg_alpha", 0, 0.5, 0.1),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 0, 1),
}


def objective(params):
    params = {
        "max_depth": int(params["max_depth"]),
        "eta": params["eta"],
        "min_child_weight": params["min_child_weight"],
        "subsample": params["subsample"],
        "reg_alpha": params["reg_alpha"],
        "colsample_bytree": params["colsample_bytree"],
        "scale_pos_weight": params["scale_pos_weight"],
    }
    xgb_hpo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        tree_method="gpu_hist",
        n_jobs=-1,
        **params,
        early_stopping_rounds=100
    )
    best_score = cross_val_score(
        xgb_hpo, X_tr_sel, y_train, scoring="neg_mean_squared_error", cv=10
    ).mean()
    loss = 1 - best_score
    return loss
