import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import src.dda.descriptive_statistics as dstat

def test_concat_skew_kurt_result_should_return_as_expected():
    data = {
        "X1": [1, 1, 1, 1],
        "X2": [2, 2, 2, 2]
    }

    df = pd.DataFrame(data)
    df_descript = df.describe()
    df_descript.loc['skew'] = df.skew()
    df_descript.loc['kurt'] = df.kurt()
    
    expected = {
        "X1": [4.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "X2": [4.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
    }
    index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt']

    df_expected = pd.DataFrame(expected, index = index)
    assert_frame_equal(df_descript, df_expected)
