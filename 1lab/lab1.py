# 1 задание
import mglearn as mgg
import matplotlib.pyplot as plt

import sys
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
import pandas as pd

print("версия Python: {}".format(sys.version))
print("версия matplotlib: {}".format(matplotlib.__version__))
print("версия NumPy: {}".format(np.__version__))
print("версия SciPy: {}".format(sp.__version__))
print("версия IPython: {}".format(IPython.__version__))
print("версия scikit-learn: {}".format(sklearn.__version__))
print("версия pandas: {}".format(pd.__version__))

# 2 задание
from sklearn.datasets import load_iris
iris_dataset = load_iris();

print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:200] + "\n...")

print("Названия ответов: {}".format(iris_dataset['target_names']))

print("Названия признаков: \n{}".format(iris_dataset['feature_names']))

print("Тип массива data: {}".format(type(iris_dataset['data'])))

print("Форма массива data: {}".format(iris_dataset['data'].shape))

print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

print("Тип массива target: {}".format(type(iris_dataset['target'])))

print("Форма массива target: {}".format(iris_dataset['target'].shape))

# 0 - setosa, 1 - versicolor, 2 - virginica
print("Ответы:\n{}".format(iris_dataset['target']))

# эффективность д-х
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))

print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

from pandas.plotting import scatter_matrix
grr = scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8,
    cmap=mgg.cm3)

plt.show()
