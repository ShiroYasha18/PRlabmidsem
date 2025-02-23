import numpy as np
import pandas as pd
df= pd.read_csv('titanic.csv')


def shan(column):
    value_counts=column.value_counts(normalize=True)
    entropy=-np.sum(value_counts * np.log2(value_counts))
    return entropy

l =shan(df["Survived"])
print(l)