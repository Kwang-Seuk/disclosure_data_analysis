import pandas as pd
import numpy as np

ds_expected = {
    "X1": [3.0, 2.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, "NaN"],
    "X2": [3.0, 2.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, "NaN"],
}
index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt']

df_ds_expected = pd.DataFrame(ds_expected, index = index)
df_ds_expected = df_ds_expected.apply(pd.to_numeric)
df_ds_expected.dtypes


import pandas as pd
test_data = {
    "X1": [1,1,1,1],
    "X2": [2,2,2,2]
}

df = pd.DataFrame(test_data)
df_descript = df.describe()

ds_skew = df.skew()
ds_kurt = df.kurt()
df_descript.loc['skew'] = ds_skew
df_descript.loc['kurt'] = ds_kurt

df_descript
