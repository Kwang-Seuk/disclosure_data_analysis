import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from scipy.stats import describe
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib import pyplot as plt
from src.dda import (
    descriptive_statistics as dstat,
    train_test_data_prep as dprep,
)

def test_concat_skew_kurt_result_should_return_as_expected():
    data = {
        "X1": [1, 1, 1, 1],
        "X2": [2, 2, 2, 2]
    }

    df = pd.DataFrame(data)
    df_descript = dstat(df)
    
    expected = {
        "X1": [4.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "X2": [4.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
    }
    index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt']

    df_expected = pd.DataFrame(expected, index = index)
    assert_frame_equal(df_descript, df_expected)

def test_descrit_groupby_result_should_return_as_expected():
    import pandas as pd
    data = {
        "Grp": ["A", "A", "B", "B"],
        "X1": [1, 1, 1, 1],
        "X2": [2, 2, 2, 2]
    }

    df = pd.DataFrame(data)
    df_grp_descript = df.groupby('Grp').agg(['mean', 'std', 'min', 'max', 'skew']).unstack()
    df_grp_descript

## Not used (just for excersize)
def test_train_test_prep_should_result_as_expected():
    data = {
        "Yr_disclusure": [2019, 2020, 2021],
        "Var1": [1, 1, 1],
        "Var2": [2, 2, 2]
    }

    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    X_train = X[X["Yr_disclusure"] < 2021]

    expected_X_train = {
        "Yr_disclusure": [2019, 2020],
        "Var1": [1, 1]
    }

    df_expected_X_train = pd.DataFrame(expected_X_train) 
    assert_frame_equal(X_train, df_expected_X_train)

def test_train_test_prep_should_result_four_dfs_as_expected():
    data = {
        "Yr_disclosure": [2019, 2020, 2021],
        "Var1": [1, 1, 1],
        "Var2": [2, 2, 2]
    }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = dprep(df, "Yr_disclosure", 2021)
    
    expected_X_train = {
        "Yr_disclosure": [2019, 2020],
        "Var1": [1, 1]
    }

    expected_X_test = {
        "Yr_disclosure": [2021],
        "Var1": [1]
    }

    expected_y_train = {
        "Var2": [2, 2]
    }

    #expected_y_test = {
    #    "Var2": [2]
    #}

    df_expected_X_train = pd.DataFrame(expected_X_train)
    df_expected_X_test = pd.DataFrame(expected_X_test)
    df_expected_y_train = pd.Series(expected_y_train)
    #df_expected_y_test = pd.DataFrame(expected_y_test)
    
    assert_frame_equal(X_train, df_expected_X_train)
    assert_frame_equal(X_test, df_expected_X_test)
    assert_series_equal(y_train, df_expected_y_train)
    #assert_frame_equal(y_train, df_expected_y_test)
