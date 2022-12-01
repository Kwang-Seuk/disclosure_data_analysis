## Loading modules
import sys
sys.path.append("..")

# Data manipulation modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from pandas.core.frame import DataFrame

# ML data preprecessing modules
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from BorutaShap import BorutaShap
from sklearn.base import clone

# XGBoost development moduels
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, plot_tree
from src.dda import (
    compute_vif_for_X,
    min_max_linspace_for_mip,
    create_df_mip_with_means_and_itp_data,
    run_mip_analysis_with_df_mip,
    plot_mip_analysis_results
)


## Load data

data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/lms_data_2020/"
data_file = "Input_rawdata_noNaN.csv"
df = pd.read_csv(data_dir + data_file)

data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/employment_rate/"
data_file = "rawdata_analysis_employment.csv"
df = pd.read_csv(data_dir + data_file, index_col=0)

## Data preparation for modelling

# Data preparation for model development
df_model = df.copy()
df_model = df.drop(["Male", "Female", "SN"], axis=1)

df_model = df.drop(["Yr_disclosure", "Regions", "Codes"], axis=1)
#"Number_of_StudR", "Number_of_StudE"
y = df_model["Credits"]
X = df_model.drop(["Credits"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True
)
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()


## Preliminary model creation and input feature seleciton (BorutaShap)

xgbm_pre = XGBRegressor(
    objective="reg:squarederror",
    max_depth=10,
    tree_method="gpu_hist",
)

Feature_Selector = BorutaShap(
    model=xgbm_pre,
    importance_measure="shap",
    classification=False,
    percentile=100,
    pvalue=0.05,
)

Feature_Selector.fit(
    X=X_train,
    y=y_train,
    n_trials=100,
    sample=False,
    train_or_test="train",
    normalize=True,
    verbose=False
)


features_to_remove = Feature_Selector.features_to_remove
X_train_boruta_shap = X_train.drop(columns=features_to_remove)
X_test_boruta_shap = X_test.drop(columns=features_to_remove)
print(len(X_train_boruta_shap.columns))

## Hyper-parameters optimization

hpspace = {
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "gamma": hp.uniform("gamma", 1, 9),
    "reg_alpha": hp.quniform("reg_alpha", 0, 0.5, 0.1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "eta": hp.uniform("eta", 0.01, 0.2),
    "min_child_weight": hp.quniform("min_child_weight", 0, 2, 0.1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 0.1, 1),
}

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
        xgb_hpo, X_train, y_train, scoring="neg_mean_squared_error", cv=10
    ).mean()
    loss = 1 - best_score
    return loss

best = fmin(
    fn=hyper_parameters_objective,
    space=hpspace,
    max_evals=50,
    rstate=np.random.default_rng(777),
    algo=tpe.suggest,
)
print(best)

## Production model development with selected input features
## and optimzied hyper-parameters setting

for key, value in best.items():
    best['max_depth']=int(best['max_depth'])
    #best['n_estimators']=int(best['n_estimators'])

xgb_model_production = XGBRegressor(
    objective ='reg:squarederror',
    n_estimators=1000,
    n_jobs = -1,
    **best,
    tree_method = "gpu_hist")    

xgb_model_production.fit(
    X_train_boruta_shap,
    y_train,
    eval_set=[(X_train_boruta_shap, y_train), (X_test_boruta_shap, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    #early_stopping_rounds=400,
)




## Most Influencing Parameters (MIP) analysis

df_interpolated = min_max_linspace_for_mip(X_train_boruta_shap, 21)
df_mip = create_df_mip_with_means_and_itp_data(X_train_boruta_shap, df_interpolated)
df_mip_input, df_mip_res = run_mip_analysis_with_df_mip(df_mip, xgb_model_production, data_dir)
plot_mip_analysis_results(df_mip_input, df_mip_res)



## Modelling resuts

# Best tree illustration
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams['figure.dpi'] = 600 
plot_tree(xgb_model_production, num_trees=xgb_model_production.get_booster().best_iteration)
plt.savefig("test_tree_stud_dropout_rate.png", dpi='figure')
#plt.show()

# Input feature selection result
Feature_Selector.plot(figsize=(30, 10), which_features="all")
plt.savefig("input_feature_selection_stud_dropout_rate.png", dpi='figure')

# Prediction results (train / test data)
tr_pred = xgb_model_production.predict(X_train_boruta_shap)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_train, tr_pred)

tst_pred = xgb_model_production.predict(X_test_boruta_shap)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_test, tst_pred)

plt.savefig("prediction_results_stud_dropout_rate.png", dpi='figure')

# Feature importance test for the selected input features
xgb_model_production.feature_importances_
feature_names = np.array(X_train_boruta_shap.columns)
sorted_idx = xgb_model_production.feature_importances_.argsort()
plt.figure(figsize=(15, 10))
plt.barh(
    feature_names[sorted_idx], xgb_model_production.feature_importances_[sorted_idx]
)
plt.xlabel("XGBoost Feature Importance")
plt.savefig("feature_importance_stud_dropout_rate.png", dpi = 'figure')

# Feature importance (permutation) test for the selected input features
perm_importance = permutation_importance(xgb_model_production, X_train_boruta_shap, y_train)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(15, 10))
plt.barh(
    feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx]
)
plt.xlabel("Permutation Importance")
plt.savefig("feature_importance_permuted_stud_dropout_rate.png", dpi = 'figure')



data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/employment_rate/"
input_mip_data_file = "employment_mip_horizontal_input.csv"
res_mip_data_file = "employment_mip_horizontal_res.csv"
mip_input_df = pd.read_csv(data_dir + input_mip_data_file, index_col=0)
mip_res_df = pd.read_csv(data_dir + res_mip_data_file, index_col = 0)

no_input_feat = len(mip_input_df.columns)

params = {
    "font.size": 13.0,
    "figure.figsize": (15, 10),
    "axes.grid": True,
    "figure.dpi": 75
}
plt.rcParams.update(params)

col_list = list(mip_input_df.columns)
subplot_titles = ["{}".format(col) for col in col_list]

fig = plt.figure()
for i in range(no_input_feat):
    ax = fig.add_subplot(5, 5, 1 + i, sharey = ax)
    ax.plot(mip_input_df.iloc[:, i], mip_res_df.iloc[:, i], 'r-')
    ax.set_title(subplot_titles[i])
fig.tight_layout()



fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(5, 5, 1 + i)
    ax.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')


    #"figure.dpi": 300









## The below is miscellenous or not used at the moment.

asciixgbm_pre.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    early_stopping_rounds=400,
)

plt.rcParams["figure.figsize"] = (15, 15)
plot_tree(xgbm_pre, num_trees=xgbm_pre.get_booster().best_iteration)
plt.show()



## Forward input feature selection test (hereafter new XGB created)
## (This code section can be used in future.)

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
fig = plot_sequential_feature_selection(
    sfs_res.get_metric_dict(), kind="std_dev"
)
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
xgbm_pre= XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    gamma=1,
    min_child_weight=1,
    colsample_bytree=1,
    max_depth=5,
)

xgbm_sel_if.fit(
    X_train_boruta_shap,
    y_train,
    eval_set=[(X_train_boruta_shap, y_train), (X_test_boruta_shap, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    early_stopping_rounds=400,
)

plt.rcParams["figure.figsize"] = (50, 50)
plot_tree(xgbm_sel_if, num_trees=xgbm_sel_if.get_booster().best_iteration)
plt.show()

## Prediction results (train / test data)

tr_sel_pred = xgbm_sel_if.predict(X_train_boruta_shap)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_train, tr_sel_pred)

tst_sel_pred = xgbm_sel_if.predict(X_test_boruta_shap)
plt.rcParams["figure.figsize"] = (15, 15)
plt.scatter(y_test, tst_sel_pred)
