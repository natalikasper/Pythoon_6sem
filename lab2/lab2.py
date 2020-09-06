import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# ------------------ЛАБА 2-------------
# запомин.набор обуч.д-х
# чтобы прогноз новый т => нах.т.в обуч.наборе, кот ближе всего к новой
# присваиваем метку этой точке для нашей новой точки
# k - м.рассм.любой число соседей и прогноз для т.д-х, кот.принадл.большинство соседей
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

print(knn.fit(X_train, y_train))

# к какому сорту привести цветок? => поместить д-е в массив numpy, вычислить форму массива
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))

# правильно?
y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))

print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

# => м.применить нашу модель (с пом.метода fit построенную) к новым д-м
