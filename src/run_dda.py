## Loading modules

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
    load_your_data,
    compute_vif_for_X,
    min_max_linspace_for_mip,
    create_df_mip_with_means_and_itp_data,
    run_mip_analysis_with_df_mip,
    plot_mip_analysis_results,
    minmax_table,
    rdn_simul_data_create,
    feat_selec_with_borutashap,
)


## Load data and data preparation for modelling

# This section allows you to load your rawdata and to produce input
# (X) and output (y) data sets. Each data sets will be divided into
# training and testing data.


# Load your data
data_dir = "/home/kjeong/kj_python/rawdata/disclosure_data/employment_rate/"
data_file = "rawdata_analysis_employment.csv"

df = pd.read_csv(data_dir + data_file, index_col = [0, 1, 2, 3])

X_train, X_test, y_train, y_test = load_your_data(df, 268, "Employment_rates")


## Input feature seleciton (BorutaShap) & hyper-parameter optimization

# This section privides BorutaShap input feature selection and
# Bayesian-based hyper-parameters optimization functions.


# Input feature selection: using BorutaShap module

X_train_boruta_shap, X_test_boruta_shap = feat_selec_with_borutashap(X_train, X_test, y_train)


# Hyper-parameters optimization

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
        xgb_hpo, X_train_boruta_shap, y_train, scoring="neg_mean_squared_error", cv=10
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


## Random simulation for input features

minmax_df, df_rdn = minmax_table(X_train_boruta_shap, 11)
rdn_simul_data_create(X_train_boruta_shap, minmax_df)



## Depicting modelling results (general figures)

# Best tree illustration
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams['figure.dpi'] = 150 
plot_tree(xgb_model_production, num_trees=xgb_model_production.get_booster().best_iteration)
plt.savefig("test_tree_stud_dropout_rate.png", dpi='figure')
#plt.show()

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


