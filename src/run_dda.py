# Loading modules
#   The following codes allow you to load necessary modules for
#   the analysis.

# Data manipulation modules
import pandas as pd
import numpy as np

# Model development modules
from hyperopt import hp, fmin, tpe
from src.dda import (
    load_your_data,
    feat_selec_with_borutashap,
    hyper_parameters_objective,
    develop_your_production_model,
    best_tree_illustration,
    feat_importance_general,
    feat_importance_permut,
    min_max_linspace_for_mip,
    create_df_mip_with_means_and_itp_data,
    run_mip_analysis_with_df_mip,
    plot_mip_analysis_results,
    minmax_table,
    rdn_simul_data_create,
    predict_plot_train_test,
)


# Load data and data preparation for modelling
#    The functions below load your input data and slit it into
#    four subsets of data: i.e. input/output and training/testing.

data_dir = "src/"
data_file = "rawdata_analysis_employment.csv"

df = pd.read_csv(data_dir + data_file, index_col=[0, 1, 2, 3])
X_train, X_test, y_train, y_test = load_your_data(df, 268, "Employment_rates")


# Input feature seleciton (BorutaShap) & hyper-parameter optimization
#    The data prepared above flow to the next step here. Tho following
#    functions distinquish input features that affect more greatly
#    to predictability from redundant features. Then the selected
#    input features are used to optimize XGBoost model hyper-parameters.

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
    "X_test": X_test_boruta_shap,
    "y_test": y_test,
}

best = fmin(
    fn=hyper_parameters_objective,
    space=hpspace,
    max_evals=10,
    rstate=np.random.default_rng(777),
    algo=tpe.suggest,
)
print(best)


# Production model development
#    This section aims at making a new XGBoost model using previsouly
#    created "selected input feature" data and optimized hyper-parameters
#    setting.

xgb_production_model = develop_your_production_model(hpspace, best)

# Depicting modelling results
#    Here you can see the shape of best tree model consisting of
#    the ensemble models, as well as prediction results and input
#    feature importance.

best_tree_illustration(xgb_production_model, (15, 15), 150, True)
predict_plot_train_test(xgb_production_model, hpspace, True)
feat_importance_general(xgb_production_model, hpspace, True)
feat_importance_permut(xgb_production_model, hpspace, True)


# Simulation (1): Most Influencing Parameters (MIP) analysis
#    The functions below execute simple simulation, to confirm
#    how the output variable respond to linear changes of one
#    selected input feature. The remaining input features are
#    fixed at average of each.

df_interpolated = min_max_linspace_for_mip(X_train_boruta_shap, 21)
df_mip = create_df_mip_with_means_and_itp_data(
    X_train_boruta_shap, df_interpolated
)
df_mip_input, df_mip_res = run_mip_analysis_with_df_mip(
    df_mip, xgb_production_model, data_dir
)
plot_mip_analysis_results(df_mip_input, df_mip_res)

## Random simulation for input features

minmax_df, df_rdn = minmax_table(X_train_boruta_shap, 11)
rdn_simul_data_create(X_train_boruta_shap, minmax_df)
