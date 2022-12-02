import sys

sys.path.append("..")

import pandas as pd
import numpy as np
import json
from scipy.stats import describe, kurtosis
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib import pyplot as plt
from src.dda import (
    descriptive_statistics_groupby as dstatg,
    create_df_mip_with_means_and_itp_data,
    min_max_linspace_for_mip,
)


def test_descript_groupby_result_should_return_as_expected():
    data = {
        "Grp": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "X1": [1, 2, 3, 4, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)
    df_grp_stats = dstatg(df, "Grp")

    result = {
        "name": "null",
        "index": [
            ["X1", "mean", "A"],
            ["X1", "mean", "B"],
            ["X1", "std", "A"],
            ["X1", "std", "B"],
            ["X1", "min", "A"],
            ["X1", "min", "B"],
            ["X1", "max", "A"],
            ["X1", "max", "B"],
            ["X1", "skew", "A"],
            ["X1", "skew", "B"],
            ["X1", "kurtosis", "A"],
            ["X1", "kurtosis", "B"],
        ],
        "data": [
            2.5,
            8.5,
            1.2909944487,
            1.2909944487,
            1.0,
            7.0,
            4.0,
            10.0,
            0.0,
            0.0,
            -1.36,
            -1.36,
        ],
    }

    result_2_df = pd.DataFrame(result_2)
    result_2_df


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
    data = {"X1": [1, 2, 3], "X2": [7, 8, 9]}

    df = pd.DataFrame(data)
    df_mean = pd.DataFrame(df.mean())
    df_mean_transposed = df_mean.transpose().reset_index(drop=True)
    df_mean_transposed

    expected = {"X1": [2.0], "X2": [8.0]}
    df_expected = pd.DataFrame(expected)
    assert_frame_equal(df_mean_transposed, df_expected)


def test_create_n_rows_df_filled_with_colname_means_from_df_should_result_as_expected():
    data = {"X1": [1, 1, 1], "X2": [2, 2, 2], "X3": [3, 3, 3]}

    df = pd.DataFrame(data)
    df_mean = pd.DataFrame(df.mean(), columns=["mean"])

    index_n = len(df.columns)
    column_titles = list(df.columns)

    df_mip = pd.DataFrame(columns=list(df.columns), index=range(len(df)))
    for col_name in df_mean.index:
        df_mip[col_name] = df_mean["mean"][col_name]

    df_mip

    expected = {
        "X1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "X2": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "X3": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    }
    df_expected = pd.DataFrame(expected)

    assert_frame_equal(df_mip, df_expected)


def test_linear_interpolation_between_min_max_should_result_as_expected():
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
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.00],
        "X2": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "X3": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
    }

    df_expected_interval_twenty = pd.DataFrame(expected_interval_twenty)
    assert_frame_equal(df_interpolated_twenty, df_expected_interval_twenty)


def test_interpolated_data_replacement_should_result_as_expected():
    data = {"X1": [1, 1, 1, 1, 1, 1], "X2": [2, 2, 2, 2, 2, 2]}
    df = pd.DataFrame(data)

    interpolated = {"X1": [1, 2, 3], "X2": [7, 8, 9]}
    df_interpolated = pd.DataFrame(interpolated)

    df_mip = df.copy()
    interpolated_length = len(df_interpolated)

    for j in range(0, len(df_mip.columns) - 1):
        for i in range(0, interpolated_length):
            df_mip.iloc[
                i : i + interpolated_length - 1, j
            ] = df_interpolated.iloc[:, j]
            i += interpolated_length
        j += 1

    expected = {"X1": [1, 2, 3, 1, 1, 1], "X2": [2, 2, 2, 7, 8, 9]}
    df_expected = pd.DataFrame(expected)

    assert_frame_equal(df_mip, df_expected)


def test_creating_mip_dataframe_should_result_as_expected():
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
