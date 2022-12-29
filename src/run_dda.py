## Loading modules

# Data manipulation modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML data preprecessing modules
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
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
    develop_your_production_model,
    best_tree_illustration,
    hyper_parameters_objective,
)


## Load data and data preparation for modelling

data_dir = "src/"
data_file = "rawdata_analysis_employment.csv"

df = pd.read_csv(data_dir + data_file, index_col=[0, 1, 2, 3])
X_train, X_test, y_train, y_test = load_your_data(df, 268, "Employment_rates")


## Input feature seleciton (BorutaShap) & hyper-parameter optimization

# Input feature selection: using BorutaShap module

X_train_boruta_shap, X_test_boruta_shap = feat_selec_with_borutashap(
    X_train, X_test, y_train
)


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
    "X_train": X_train_boruta_shap,
    "y_train": y_train,
}

best = fmin(
    fn=hyper_parameters_objective,
    space=hpspace,
    max_evals=5,
    rstate=np.random.default_rng(777),
    algo=tpe.suggest,
)
print(best)


## Production model development with selected input features
## and optimzied hyper-parameters setting

xgb_production_model = develop_your_production_model(
    X_train_boruta_shap, y_train, X_test_boruta_shap, y_test, best
)

for key, value in best.items():
    best["max_depth"] = int(best["max_depth"])
    # best['n_estimators']=int(best['n_estimators'])

xgb_model_production = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    n_jobs=-1,
    **best,
    tree_method="gpu_hist"
)

xgb_model_production.fit(
    X_train_boruta_shap,
    y_train,
    eval_set=[(X_train_boruta_shap, y_train), (X_test_boruta_shap, y_test)],
    eval_metric=["rmse"],
    verbose=100,
    # early_stopping_rounds=400,
)

## Most Influencing Parameters (MIP) analysis

df_interpolated = min_max_linspace_for_mip(X_train_boruta_shap, 21)
df_mip = create_df_mip_with_means_and_itp_data(
    X_train_boruta_shap, df_interpolated
)
df_mip_input, df_mip_res = run_mip_analysis_with_df_mip(
    df_mip, xgb_model_production, data_dir
)
plot_mip_analysis_results(df_mip_input, df_mip_res)

## Random simulation for input features

minmax_df, df_rdn = minmax_table(X_train_boruta_shap, 11)
rdn_simul_data_create(X_train_boruta_shap, minmax_df)

## Depicting modelling results

best_tree_illustration(xgb_production_model, (15, 15), 150, True)
predict_train_test(
    xgb_production_model,
    X_train_boruta_shap,
    y_train,
    X_test_boruta_shap,
    y_test,
    True,
)
feat_importance_general(xgb_production_model, X_train_boruta_shap)
feat_importance_permut(xgb_production_model, X_train_boruta_shap, y_train)

import os

cwd = os.getcwd()
print(cwd)
