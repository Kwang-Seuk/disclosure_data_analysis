import os
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
from src.dda import (
    load_your_data,
    feat_selec_with_borutashap,
    create_interpolation_for_mip,
    create_df_means_for_mip,
    create_random_dataframes,
)


@pytest.fixture(scope="module")
def fixture_input_df():
    x1 = np.random.standard_normal(1000)
    x2 = np.random.standard_normal(1000) * 1.5
    x3 = [i * i for i in x2]
    x4 = [i / j for i, j in zip(x1, x2)]
    x5 = [i * j for i, j in zip(x1, x2)]
    y = [i / (i * j) for i, j in zip(x1, x2)]

    df = pd.DataFrame(
        list(zip(x1, x2, x3, x4, x5, y)),
        columns=["x1", "x2", "x3", "x4", "x5", "y"],
    )

    return df


@pytest.fixture
def df_interpolated():
    data = {
        "column1": [1, 2, 3, 4, 5],
        "column2": [5, 6, 7, 8, 9],
        "column3": [3, 4, 5, 6, 7],
    }
    return pd.DataFrame(data)


def test_data_loading_should_return_expected():
    data = {
        "x1": [1, 1, 1, 10, 10],
        "x2": [2, 2, 2, 20, 20],
        "y": [3, 3, 3, 30, 30],
    }

    df = pd.DataFrame(data)
    x_train, x_test, y_train, y_test = load_your_data(df, 3, "y")

    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    expected_x_train = {
        "x1": [1, 1, 1],
        "x2": [2, 2, 2],
    }
    expected_x_test = {
        "x1": [10, 10],
        "x2": [20, 20],
    }

    df_expected_x_train = pd.DataFrame(expected_x_train)
    df_expected_x_test = pd.DataFrame(expected_x_test)

    ser_expected_y_train = pd.Series([3, 3, 3], dtype="int64", name="y")
    ser_expected_y_test = pd.Series([30, 30], dtype="int64", name="y")

    assert_frame_equal(x_train, df_expected_x_train)
    assert_frame_equal(x_test, df_expected_x_test)

    assert_series_equal(y_train, ser_expected_y_train)
    assert_series_equal(y_test, ser_expected_y_test)


def test_selected_feats_no_should_be_smaller_than_raw_input_feats(
    fixture_input_df,
):
    x_train, x_test, y_train, y_test = load_your_data(
        fixture_input_df, 600, "y"
    )

    x_selec_train, x_selec_test = feat_selec_with_borutashap(
        x_train, x_test, y_train, 50, False
    )

    assert len(x_train.columns) >= len(x_selec_train.columns)
    assert len(x_test.columns) >= len(x_selec_test.columns)


def test_interpolation_for_mip_should_result_as_expected():
    # This test is for src.dda.min_max_linspace_for_mip() running.
    data = {
        "x_train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)
    input_dict = {"x_train": df}

    df_itp_four = create_interpolation_for_mip(input_dict, 4)

    expected_interval_four = {
        "x_train": [1.0, 4.0, 7.0, 10.0],
    }
    df_expected_interval_four = pd.DataFrame(expected_interval_four)
    assert_frame_equal(df_itp_four, df_expected_interval_four)

    df_interpolated_twenty = create_interpolation_for_mip(input_dict, 20)
    expected_interval_twenty = {
        "x_train": [
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
    }

    df_expected_interval_twenty = pd.DataFrame(expected_interval_twenty)
    assert_frame_equal(df_interpolated_twenty, df_expected_interval_twenty)


def test_create_means_for_mip_should_result_as_expected():
    data_x_train = {
        "X1": [1, 2, 3, 4, 5, 6],
        "X2": [4, 5, 6, 7, 8, 9],
        "X3": [7, 8, 9, 6, 7, 8],
    }
    x_train_test = pd.DataFrame(data_x_train)
    input_dict = {"x_train": x_train_test}

    df_itp = create_interpolation_for_mip(input_dict, 3)
    df_means = create_df_means_for_mip(input_dict, df_itp)

    expected_itp_interval_three = {
        "X1": [1.0, 3.5, 6.0],
        "X2": [4.0, 6.5, 9.0],
        "X3": [6.0, 7.5, 9.0],
    }
    df_expected_itp = pd.DataFrame(expected_itp_interval_three)

    expected_means = {
        "X1": [3.5, 3.5, 3.5],
        "X2": [6.5, 6.5, 6.5],
        "X3": [7.5, 7.5, 7.5],
    }
    df_expected_means = pd.DataFrame(expected_means)

    assert_frame_equal(df_itp, df_expected_itp)
    assert_frame_equal(df_means, df_expected_means)


def test_create_random_dataframes(df_interpolated):
    n = 10000
    create_random_dataframes(df_interpolated, n=n)

    for col in df_interpolated.columns:
        filename = f"df_randomized_{col}.csv"
        assert os.path.exists(filename)

        df = pd.read_csv(filename)
        assert df.shape == (
            n * df_interpolated.shape[0],
            len(df_interpolated.columns),
        )
        assert np.all(
            df[col].values == np.repeat(df_interpolated[col].values, n)
        )

        os.remove(filename)
