import mglearn
import mglearn.plots as mp
import matplotlib.pyplot as plt
import numpy as np

# подсчитаем частоты:
from sklearn.linear_model import LinearRegression
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

# подсчет ненулевых эл-тов в к.классе:
counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))


# ДЕРЕВЬЯ РЕШЕНИЙ (иерархия правил)
mp.plot_animal_tree()
# вместоручного - с пом.контрол.обуч.
plt.show()


# ПОСТРОЕНИЕ ДЕРЕВЬЕВ РЕШЕНИЙ
# для набора д-х two_moons (2 класса, к.т.-маркер)
mglearn.plots.plot_tree_progressive()
plt.show()


# ПРЕДВАРИТЕЛЬНАЯ ОБРЕЗКА
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# выращиваем дерево пока все листья не станут чистыми (для воспроизводимости рез-тов)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))

# применим (остан.проц. постр.дерева до того как мы идеально подгоним модель к обуч.д-м)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(tree.score(X_test, y_test)))


# АНАЛИЗ ДЕРЕВЬЕВ РЕШЕНИЙ
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# ВАЖНОСТЬ ПРИЗНАКОВ В ДЕРЕВЬЯХ
print("Важности признаков:\n{}".format(tree.feature_importances_))

for name, score in zip(cancer["feature_names"], tree.feature_importances_):
    print(name, score)

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
plot_feature_importances_cancer(tree)
plt.show()

# экстраполировать
import pandas as pd
ram_prices = pd.read_csv("F:/ram_price.csv")
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Год")
plt.ylabel("Цена $/Мбайт")
plt.show()

# сравним DecisionTreeRegressor и LinearRegression
from sklearn.tree import DecisionTreeRegressor
# используем исторические данные для прогнозирования цен после 2000 года
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:, np.newaxis]

y_train = np.log(data_train.price)
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
# прогнозируем по всем данным
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_test.date, data_test.price, label="Тестовые данные")
plt.semilogy(ram_prices.date, price_tree, label="Прогнозы дерева")
plt.semilogy(ram_prices.date, price_lr, label="Прогнозы линейной регрессии")
plt.legend()
plt.show()
