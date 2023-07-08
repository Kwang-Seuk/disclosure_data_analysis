# Loading modules
from typing import Tuple, Dict, Union
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from xgboost import XGBRegressor
from hyperopt import hp
from src.dda import (
    load_and_check_data,
    prepare_train_test_data,
    feat_selec_with_borutashap,
    hyper_parameters_tuning,
    develop_production_model,
    production_model_rmse_display,
    best_tree_illustration,
    feat_importance_general,
    feat_importance_permut,
    create_interpolation_for_mip,
    create_df_means_for_mip,
    create_random_dataframes,
    mip_analysis,
    plot_mip_analysis_results,
    predict_plot_train_test,
    random_simulation,
    draw_scatter_graphs_from_csv,
    draw_contour_graphs_from_csv,
)
from src.validate import (
    validate_train_test_data,
    validate_selected_features,
    validate_production_model,
    validate_df_interpolated,
    validate_df_means,
    validate_mip_analysis,
    validate_create_random_dataframes,
    validate_model_loading,
    validate_saved_csv,
)

# Preparation for modelling
data_dir = "src/"
data_file = "RStuR_Enrol.csv"
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
}

# Pipeline


def step1_prepare_modelling_data(
    data_dir: str, data_file: str
) -> Tuple[DataFrame, DataFrame, Series, Series]:
    df = load_and_check_data(data_dir, data_file)
    x_train, x_test, y_train, y_test = prepare_train_test_data(
        df, 268, "REnrol_StuR"
    )
    validate_train_test_data(x_train, x_test, y_train, y_test)
    return (x_train, x_test, y_train, y_test)


def step2_develop_your_model(
    x_train: DataFrame,
    x_test: DataFrame,
    y_train: Series,
    y_test: Series,
    hpspace: Dict,
    gpu_flag: bool,
) -> Tuple[XGBRegressor, DataFrame, DataFrame]:
    x_train_boruta_shap, x_test_boruta_shap = feat_selec_with_borutashap(
        x_train, x_test, y_train, 50, gpu_flag
    )
    validate_selected_features(x_train_boruta_shap, x_test_boruta_shap)
    best_hyperparams = hyper_parameters_tuning(
        hpspace, x_train_boruta_shap, y_train
    )
    xgb_production_model = develop_production_model(
        (x_train_boruta_shap, y_train),
        (x_test_boruta_shap, y_test),
        best_hyperparams,
        10000,
        gpu_flag,
    )
    production_model_rmse_display(xgb_production_model)
    validate_production_model(xgb_production_model)

    return xgb_production_model, x_train_boruta_shap, x_test_boruta_shap


def step3_model_performance_test(
    model: XGBRegressor,
    train_data: Tuple[DataFrame, Series],
    test_data: Tuple[DataFrame, Series],
    fig_dpi: int = 300,
    fig_save: bool = False,
) -> None:
    best_tree_illustration(model, fig_dpi, fig_save)
    predict_plot_train_test(model, train_data, test_data, fig_dpi, fig_save)
    feat_importance_general(model, train_data, fig_dpi, fig_save)
    feat_importance_permut(model, train_data, fig_dpi, fig_save)


def step4_mip_analysis(
    x_train: DataFrame,
    model: XGBRegressor,
    interval: int = 11,
    fig_dpi: int = 300,
    fig_size: Tuple = (10, 10),
    save_res: bool = True,
) -> None:
    df_itp = create_interpolation_for_mip(x_train, interval)
    validate_df_interpolated(df_itp, x_train, interval)

    df_means = create_df_means_for_mip(x_train, df_itp)
    validate_df_means(df_means, x_train, df_itp)

    df_mip_res = mip_analysis(df_means, df_itp, model, save_res)
    validate_mip_analysis(df_mip_res, df_means, df_itp)

    plot_mip_analysis_results(df_itp, df_mip_res, fig_dpi, fig_size, save_res)


def step5_random_simulation(
    input_csv: str,
    model: Union[XGBRegressor, str],
    n: int,
    df_itp_employ: DataFrame,
    control_var: str,
    x_var: str,
    y_var: str,
) -> None:
    create_random_dataframes(df_itp_employ, n)
    validate_create_random_dataframes(df_itp_employ, n)

    validate_model_loading(model)
    df_rand = random_simulation(input_csv, model)
    validate_saved_csv(input_csv, df_rand)

    draw_scatter_graphs_from_csv(
        "random_simulation_res_" + input_csv, control_var, x_var, y_var
    )
    draw_contour_graphs_from_csv(
        "random_simulation_res_" + input_csv, control_var, x_var, y_var
    )


def main_pipeline(
    data_dir: str, data_file: str, hpspace: Dict, gpu_flag: bool
) -> None:
    # Step 1: Prepare modelling data
    x_train, x_test, y_train, y_test = step1_prepare_modelling_data(
        data_dir, data_file
    )

    # Step 2: Develop your model
    (
        xgb_production_model,
        x_train_boruta_shap,
        x_test_boruta_shap,
    ) = step2_develop_your_model(
        x_train, x_test, y_train, y_test, hpspace, gpu_flag=False
    )

    # Step 3: Model performance test
    step3_model_performance_test(
        xgb_production_model,
        (x_train_boruta_shap, y_train),
        (x_test_boruta_shap, y_test),
    )

    # Step 4: MIP analysis
    step4_mip_analysis(x_train_boruta_shap, xgb_production_model)

    # Step 5: Random simulation
    step5_random_simulation(
        input_csv,
        xgb_production_model,
        10,
        x_train_boruta_shap,
        "control_var",
        "x_var",
        "y_var",
    )


if __name__ == "__main__":
    main_pipeline()
