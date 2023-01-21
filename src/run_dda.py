# Loading modules
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe
from src.dda import (
    load_your_data,
    feat_selec_with_borutashap,
    hyper_parameters_objective,
    develop_production_model,
    production_model_rmse_display,
    best_tree_illustration,
    feat_importance_general,
    feat_importance_permut,
    create_interpolation_for_mip,
    create_df_means_for_mip,
    mip_analysis,
    plot_mip_analysis_results,
    predict_plot_train_test,
)


# Load data and input feature selection for modelling
data_dir = "src/"
data_file = "rawdata_analysis_employment.csv"

df = pd.read_csv(data_dir + data_file, index_col=[0, 1, 2, 3])
x_train, x_test, y_train, y_test = load_your_data(df, 268, "Employment_rates")

x_train_boruta_shap, x_test_boruta_shap = feat_selec_with_borutashap(
    x_train, x_test, y_train, 50
)

# Hyper-parameters optimization
hpspace = {
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "gamma": hp.uniform("gamma", 0, 10),
    "reg_alpha": hp.quniform("reg_alpha", 0, 0.5, 0.1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "eta": hp.uniform("eta", 0, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 2, 0.1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 0.1, 1),
    "x_train": x_train_boruta_shap,
    "y_train": y_train,
}

best = fmin(
    fn=hyper_parameters_objective,
    space=hpspace,
    max_evals=50,
    rstate=np.random.default_rng(777),
    algo=tpe.suggest,
)
print(best)

# Production model development
input_data_dict = {
    "x_train": x_train_boruta_shap,
    "x_test": x_test_boruta_shap,
    "y_train": y_train,
    "y_test": y_test,
}
xgb_production_model = develop_production_model(input_data_dict, 10000, best)
production_model_rmse_display(xgb_production_model)

# Depicting modelling results
best_tree_illustration(xgb_production_model, 150, False)
predict_plot_train_test(xgb_production_model, input_data_dict, 150, False)
feat_importance_general(xgb_production_model, input_data_dict, 150, False)
feat_importance_permut(xgb_production_model, input_data_dict, 150, False)


# Simulation (1): Most Influencing Parameters (MIP) analysis
df_itp_employ = create_interpolation_for_mip(input_data_dict, 21)
df_means_employ = create_df_means_for_mip(input_data_dict, df_itp_employ)
df_mip_res_employ = mip_analysis(
    df_means_employ, df_itp_employ, xgb_production_model, False
)
plot_mip_analysis_results(df_itp_employ, df_mip_res_employ, 150, False)
