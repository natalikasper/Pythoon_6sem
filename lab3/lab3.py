import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = mglearn.datasets.make_forge()

# диаграмма расс.
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"])
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
plt.show()

# регрессия
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.show()


# задача - дать проноз является ли опухоль злокач.
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Ключи cancer(): \n{}".format(cancer.keys()))

# форма - shape (т, пр)
print("Форма массива data для набора cancer: {}".format(cancer.data.shape))

# 212з, 357д
print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Имена признаков:\n{}".format(cancer.feature_names))
# print(cancer.DESCR[:193])


from sklearn.datasets import load_boston
boston = load_boston()
print("форма массива data для набора boston: {}".format(boston.data.shape))
# print(boston.DESCR[:193])

X, y = mglearn.datasets.load_extended_boston()
print("форма массива X: {}".format(X.shape))


# метод k ближ.соседей (forge)
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

# с.указать k соседей
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()


# метод ближ.соседей (реал.набор)
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

print(clf.fit(X_train, y_train))

print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))



# АНАЛИЗ KNeighborsClassifier
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
 clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("количество соседей:{}".format(n_neighbors))
 ax.set_xlabel("признак 0")
 ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
plt.show()


# ----ЕСТЬ ЛИ СВЯЗЬ МЕЖДУ СЛОЖНОСТЬЮ МОДЕЛИ И ОБОБЩ,СПОСОБНОСТЬЮ?---
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# пробуем n_neighbors от 1 до 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:

 clf = KNeighborsClassifier(n_neighbors=n_neighbors)
 clf.fit(X_train, y_train)

 training_accuracy.append(clf.score(X_train, y_train))

 test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()
plt.show()

# РЕГРЕССИЯ БЛИЖАЙШИХ СОСЕДЕЙ (1)
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

# исп-ем нес.соседей
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

# алгоритм регрессия к ближ.соседей
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)

print(reg.fit(X_train, y_train))
print("Прогнозы для тестового набора:\n{}".format(reg.predict(X_test)))

print("R^2 на тестовом наборе: {:.2f}".format(reg.score(X_test, y_test)))

# АНАЛИЗ МОДЕЛИ REGRESSOR
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):

 reg = KNeighborsRegressor(n_neighbors=n_neighbors)
 reg.fit(X_train, y_train)
 ax.plot(line, reg.predict(line))
 ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
 ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
 ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
 n_neighbors, reg.score(X_train, y_train),
 reg.score(X_test, y_test)))
 ax.set_xlabel("Признак")
 ax.set_ylabel("Целевая переменная")
axes[0].legend(["Прогнозы модели", "Обучающие данные/ответы", "Тестовые данные/ответы"], loc="best")
plt.show()
