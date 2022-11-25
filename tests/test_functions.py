import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import json
from scipy.stats import describe
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib import pyplot as plt
from src.dda import (
    descriptive_statistics as dstat,
    create_df_mip_with_means_and_itp_data,
    min_max_linspace_for_mip
)


def test_concat_skew_kurt_result_should_return_as_expected():
    data = {"X1": [1, 1, 1, 1], "X2": [2, 2, 2, 2]}

    df = pd.DataFrame(data)
    df_descript = dstat(df)
    df_descript

    expected = {
        "X1": [4.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "X2": [4.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0],
    }
    index = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "skew",
        "kurt",
    ]

    df_expected = pd.DataFrame(expected, index=index)
    assert_frame_equal(df_descript, df_expected)

def test_descript_groupby_result_should_return_as_expected():
    data = {
        "Grp": ["A", "A", "B", "B"],
        "X1": [1, 1, 1, 1],
    }

    df = pd.DataFrame(data)
    df_grp_descript = (
        df.groupby("Grp").agg(["mean", "std", "min", "max", "skew"]).unstack()
    )


def test_rawdata_csv_to_json_result_should_return_as_expected():
    csv_data = pd.read_csv(
        "/home/kjeong/kj_python/myrepos/b510/disclosure_data_analysis/tests/testcode_csv.csv",
        sep=",",
    )
    json_data = csv_data.to_json(orient="records")

    expected = (
        {"SN": 1, "X1": 1, "X2": 2},
        {"SN": 2, "X1": 1, "X2": 2},
        {"SN": 3, "X1": 1, "X2": 2},
    )

    assert json_data == expected

def test_mean_calculation_should_result_as_expected():
    data = {
        "X1": [1, 2, 3],
        "X2": [7, 8, 9]
    }

    df = pd.DataFrame(data)
    df_mean = pd.DataFrame(df.mean())
    df_mean_transposed = df_mean.transpose().reset_index(drop = True)
    df_mean_transposed

    expected = {
        "X1": [2.0],
        "X2": [8.0]
    }
    df_expected = pd.DataFrame(expected)
    assert_frame_equal(df_mean_transposed, df_expected)
    
def test_create_n_rows_df_filled_with_colname_means_from_df_should_result_as_expected():
    data = {
        "X1": [1, 1, 1],
        "X2": [2, 2, 2],
        "X3": [3, 3, 3]
    }

    df = pd.DataFrame(data)
    df_mean = pd.DataFrame(df.mean(), columns = ["mean"])
    
    index_n = len(df.columns)
    column_titles = list(df.columns)

    df_mip = pd.DataFrame(columns = list(df.columns), index = range(len(df)))
    for col_name in df_mean.index:
        df_mip[col_name] = df_mean["mean"][col_name]

    df_mip

    expected = {
        "X1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "X2": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "X3": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    }
    df_expected = pd.DataFrame(expected)

    assert_frame_equal(df_mip, df_expected)

def test_linear_interpolation_between_min_max_should_result_as_expected():
    data = {
        "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "X2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    df = pd.DataFrame(data)
    df_interpolated_four = min_max_linspace_for_mip(df, 4)

    df_interpolated_four

    expected_interval_four = {
        "X1": [1.0, 4.0, 7.0, 10.0],
        "X2": [11.0, 14.0, 17.0, 20.0]
    }

    df_expected_interval_four = pd.DataFrame(expected_interval_four)
    assert_frame_equal(df_interpolated_four, df_expected_interval_four)
    
    df_interpolated_twenty = min_max_linspace_for_mip(df, 20)
    expected_interval_twenty = {
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.00],
        "X2": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    }

    df_expected_interval_twenty = pd.DataFrame(expected_interval_twenty)
    assert_frame_equal(df_interpolated_twenty, df_expected_interval_twenty)

def test_interpolated_data_replacement_should_result_as_expected():
    data = {
        "X1": [1, 1, 1, 1, 1, 1],
        "X2": [2, 2, 2, 2, 2, 2]
    }
    df = pd.DataFrame(data)

    interpolated = {
        "X1": [1, 2, 3],
        "X2": [7, 8, 9]
    }
    df_interpolated = pd.DataFrame(interpolated)

    df_mip = df.copy()
    interpolated_length = len(df_interpolated)

    for j in range(0, len(df_mip.columns)-1):
        for i in range(0, interpolated_length):
            df_mip.iloc[i:i+interpolated_length-1, j] = df_interpolated.iloc[:, j]
            i += interpolated_length
        j += 1

    expected = {
        "X1": [1, 2, 3, 1, 1, 1],
        "X2": [2, 2, 2, 7, 8, 9]
    }
    df_expected = pd.DataFrame(expected)

    assert_frame_equal(df_mip, df_expected)

def test_replacing_a_column_data_with_another_df_column_data_should_result_as_expected():
    data = {
        "X1": [10, 20, 30],
        "X2": [70, 80, 90]
    }
    df = pd.DataFrame(data)

    df_mip = create_df_filled_with_means(df)
    for col_name in df.columns:
        df_mip_itp = df_mip.copy()
        df_mip_itp.loc[:, col_name] = df.loc[:, col_name]
        vars()["df_mip_itp_" + col_name] = df_mip_itp

    expected_X1_data = {
        "X1": [10, 20, 30],
        "X2": [80.0, 80.0, 80.0]
    }
    expected_X1_df = pd.DataFrame(expected_X1_data)
    assert_frame_equal(df_mip_itp_X1, expected_X1_df)

    expected_X2_data = {
        "X1": [20.0, 20.0, 20.0],
        "X2": [70, 80, 90]
    }
    expected_X2_df = pd.DataFrame(expected_X2_data)
    assert_frame_equal(df_mip_itp_X2, expected_X2_df)

def test_creating_mip_dataframe_should_result_as_expected():
    data = {
        "X1": [1, 2, 3, 4, 5, 6],
        "X2": [4, 5, 6, 7, 8, 9],
        "X3": [7, 8, 9, 6, 7, 8]
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
        "X3_means": [7.5, 7.5, 7.5]
    }
    df_expected = pd.DataFrame(expected)

    assert_frame_equal(df_mip, df_expected)

def test_creating_mip_input_data_by_alternating_itp_and_means_should_result_as_expected():
    mip_input_data = {
        "X1": [1, 2, 3],
        "X2": [7, 8, 9],
        "X3": [4, 5, 6],
        "X1_means": [2.5, 2.5, 2.5],
        "X2_means": [7.5, 7.5, 7.5],
        "X3_means": [5.5, 5.5, 5.5]
    }
    df_mip = pd.DataFrame(mip_input_data)

    X = df_mip.iloc[:,0]
    X_var = X.columns + "_means"
    X


    for i in range(len(df_mip) / 2 - 1):
        X = df_mip.iloc[:,i]
        X_var = X.columns

        X_means_tmp = df_mip.filer(regex = 

