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
    data = {"X1": [1, 1, 1, 1], "X2": [2, 2, 2, 2]}

    df = pd.DataFrame(data)
    df_descript = dstat(df)

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
    import pandas as pdi
    import json

    data = {
        "Grp": ["A", "A", "B", "B"],
        "X1": [1, 1, 1, 1],
    }

    df = pd.DataFrame(data)
    df_grp_descript = (
        df.groupby("Grp").agg(["mean", "std", "min", "max", "skew"]).unstack()
    )
