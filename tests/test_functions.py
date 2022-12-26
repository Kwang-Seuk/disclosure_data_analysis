import sys

sys.path.append("..")

import pandas as pd
import numpy as np
import json
from scipy.stats import describe, kurtosis
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib import pyplot as plt
from src.dda import (
    create_df_mip_with_means_and_itp_data,
    min_max_linspace_for_mip,
    minmax_table,
    rdn_simul_data_create
)


def test_create_n_rows_df_filled_with_colname_means_from_df_should_result_as_expected():
    data = {"X1": [1, 1, 1], "X2": [2, 2, 2], "X3": [3, 3, 3]}

    df = pd.DataFrame(data)
    df_mean = pd.DataFrame(df.mean(), columns=["mean"])

    index_n = len(df.columns)
    column_titles = list(df.columns)

    df_mip = pd.DataFrame(columns=list(df.columns), index=range(len(df)))
    for col_name in df_mean.index:
        df_mip[col_name] = df_mean["mean"][col_name]

    expected = {
        "X1": [1.0, 1.0, 1.0],
        "X2": [2.0, 2.0, 2.0],
        "X3": [3.0, 3.0, 3.0],
    }
    df_expected = pd.DataFrame(expected)
    assert_frame_equal(df_mip, df_expected)


def test_linear_interpolation_between_min_max_should_result_as_expected():
    
    # This test is for src.dda.min_max_linspace_for_mip() running.
    data = {
        "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "X2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "X3": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    }
    df = pd.DataFrame(data)
    df_interpolated_four = min_max_linspace_for_mip(df, 4)

    expected_interval_four = {
        "X1": [1.0, 4.0, 7.0, 10.0],
        "X2": [11.0, 14.0, 17.0, 20.0],
        "X3": [21.0, 24.0, 27.0, 30.0],
    }
    df_expected_interval_four = pd.DataFrame(expected_interval_four)
    assert_frame_equal(df_interpolated_four, df_expected_interval_four)

    df_interpolated_twenty = min_max_linspace_for_mip(df, 20)
    expected_interval_twenty = {
        "X1": [
            1.0,
            1.4736842105263157,
            1.9473684210526314,
            2.4210526315789473,
            2.894736842105263,
            3.3684210526315788,
            3.8421052631578947,
            4.315789473684211,
            4.789473684210526,
            5.263157894736842,
            5.7368421052631575,
            6.2105263157894735,
            6.684210526315789,
            7.157894736842105,
            7.63157894736842,
            8.105263157894736,
            8.578947368421051,
            9.052631578947368,
            9.526315789473683,
            10.0,
        ],
        "X2": [
            11.0,
            11.473684210526315,
            11.947368421052632,
            12.421052631578947,
            12.894736842105264,
            13.368421052631579,
            13.842105263157894,
            14.31578947368421,
            14.789473684210526,
            15.263157894736842,
            15.736842105263158,
            16.210526315789473,
            16.684210526315788,
            17.157894736842106,
            17.63157894736842,
            18.105263157894736,
            18.57894736842105,
            19.05263157894737,
            19.526315789473685,
            20.0,
        ],
        "X3": [
            21.0,
            21.473684210526315,
            21.94736842105263,
            22.42105263157895,
            22.894736842105264,
            23.36842105263158,
            23.842105263157894,
            24.31578947368421,
            24.789473684210527,
            25.263157894736842,
            25.736842105263158,
            26.210526315789473,
            26.684210526315788,
            27.157894736842106,
            27.63157894736842,
            28.105263157894736,
            28.57894736842105,
            29.05263157894737,
            29.526315789473685,
            30.0,
        ],
    }

    df_expected_interval_twenty = pd.DataFrame(expected_interval_twenty)
    assert_frame_equal(df_interpolated_twenty, df_expected_interval_twenty)


def test_creating_mip_dataframe_should_result_as_expected():

    # This test function is designed to test the implementation of
    # min_max_linspace_for_mip() and creat_df_mip_with_means_and_itp_data()
    # simultanesouly. The result dataframe is subjected to the MIP analysis
    # in modelling process
    
    data = {
        "X1": [1, 2, 3, 4, 5, 6],
        "X2": [4, 5, 6, 7, 8, 9],
        "X3": [7, 8, 9, 6, 7, 8],
    }
    df = pd.DataFrame(data)

    df_itp = min_max_linspace_for_mip(df, 3)
    df_mip = create_df_mip_with_means_and_itp_data(df, df_itp)

    expected = {
        "X1": [1.0, 3.5, 6.0],
        "X2": [4.0, 6.5, 9.0],
        "X3": [6.0, 7.5, 9.0],
        "X1_means": [3.5, 3.5, 3.5],
        "X2_means": [6.5, 6.5, 6.5],
        "X3_means": [7.5, 7.5, 7.5],
    }
    df_expected = pd.DataFrame(expected)
    assert_frame_equal(df_mip, df_expected)

def test_minmax_table_function_should_result_as_expected():

    data = {
        "X1": [1, 2, 3, 4, 5, 6],
        "X2": [4, 5, 6, 7, 8, 9],
        "X3": [7, 8, 9, 6, 7, 8],
    }
    df = pd.DataFrame(data)
    minmax_df, df_rdn = minmax_table(df, 10)

    minmax_data_expected = {
        "index": ["X1", "X2", "X3"],
        "nunique": [6, 6, 4],
        "max": [6, 9, 9], 
        "min": [1, 4, 6],
        "dtypes": ["int64", "int64", "int64"]
    }

    rdn_data_expected = {
        "X1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "X2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "X3": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    

    df_minmax_data_expected = pd.DataFrame(minmax_data_expected)
    df_minmax_data_expected.set_index('index', inplace=True)
    df_minmax_data_expected.index.name = None
    df_rdn_data_expected = pd.DataFrame(rdn_data_expected)

    assert_frame_equal(minmax_df, df_minmax_data_expected)
    assert_frame_equal(df_rdn,  df_rdn_data_expected, check_frame_type = True)

def test_random_simulation_function_should_result_as_expected():

    data = {
        "X1": [1, 2, 3, 4, 5, 6],
        "X2": [4, 5, 6, 7, 8, 9],
        "X3": [7, 8, 9, 6, 7, 8],
    }
    df = pd.DataFrame(data)

    minmax_df, df_rdn = minmax_table(df, 10)
    out_dir = "/home/kjeong/kj_python/myrepos/kjrepo/bsv/"
    rdn_simul_data_create(df, minmax_df, df_rdn, 11, 100, True, out_dir)


