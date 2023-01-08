import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from src.dda import (
    load_your_data,
    create_interpolation_for_mip,
    create_df_means_for_mip,
    minmax_table,
)


def test_data_loading_should_return_expected():

    data = {
        "x1": [1, 1, 1, 10, 10],
        "x2": [2, 2, 2, 20, 20],
        "y": [3, 3, 3, 30, 30],
    }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = load_your_data(df, 3, "y")

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    expected_X_train = {
        "x1": [1, 1, 1],
        "x2": [2, 2, 2],
    }
    expected_X_test = {
        "x1": [10, 10],
        "x2": [20, 20],
    }

    df_expected_X_train = pd.DataFrame(expected_X_train)
    df_expected_X_test = pd.DataFrame(expected_X_test)

    ser_expected_y_train = pd.Series([3, 3, 3], dtype="int64", name="y")
    ser_expected_y_test = pd.Series([30, 30], dtype="int64", name="y")

    assert_frame_equal(X_train, df_expected_X_train)
    assert_frame_equal(X_test, df_expected_X_test)

    assert_series_equal(y_train, ser_expected_y_train)
    assert_series_equal(y_test, ser_expected_y_test)


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


def test_minmax_table_function_should_result_as_expected():

    data = {
        "X1": [1, 2, 3, 4],
        "X2": [4, 5, 6, 7],
        "X3": [7, 7, 8, 8],
    }
    df = pd.DataFrame(data)
    data_dict = {"x_train": df}
    minmax_df, df_rdn = minmax_table(data_dict, 5)

    minmax_data_expected = {
        "index": ["X1", "X2", "X3"],
        "nunique": [4, 4, 2],
        "max": [4, 7, 8],
        "min": [1, 4, 7],
        "dtypes": ["int64", "int64", "int64"],
    }

    df_minmax_data_expected = pd.DataFrame(minmax_data_expected)
    df_minmax_data_expected.set_index("index", inplace=True)
    df_minmax_data_expected.index.name = None

    assert_frame_equal(minmax_df, df_minmax_data_expected)

    ser_df_rdn_count = df_rdn.count()
    expected_rdn_count = {"X1": 5, "X2": 5, "X3": 5}
    ser_expected_rdn_count = pd.Series(expected_rdn_count)

    assert_series_equal(ser_df_rdn_count, ser_expected_rdn_count)

    ser_df_rdn_min = df_rdn.min()
    expected_rdn_min = {"X1": 1, "X2": 4, "X3": 7}

    assert ser_df_rdn_min[0] >= expected_rdn_min["X1"]
    assert ser_df_rdn_min[1] >= expected_rdn_min["X2"]
    assert ser_df_rdn_min[2] >= expected_rdn_min["X3"]

    ser_df_rdn_max = df_rdn.max()
    expected_rdn_max = {"X1": 4, "X2": 7, "X3": 8}

    assert ser_df_rdn_max[0] <= expected_rdn_max["X1"]
    assert ser_df_rdn_max[1] <= expected_rdn_max["X2"]
    assert ser_df_rdn_max[2] <= expected_rdn_max["X3"]
