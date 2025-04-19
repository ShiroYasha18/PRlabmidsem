import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris= load_iris()
df= pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']= iris.target

def sha(column):
    value_count= column.value_counts(normalize=True)
    entropy = -np.sum ( value_count*np.log2(value_count))
    return entropy

s= sha(df['target'])
print(s)

