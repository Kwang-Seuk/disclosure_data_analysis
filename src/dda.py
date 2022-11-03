import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

def descriptive_statistics(df: DataFrame) -> DataFrame:
    df_descript = df.describe()
    df_descript.loc["skew"] = df.skew()
    df_descript.loc["kurt"] = df.kurt()
    return df_descript

