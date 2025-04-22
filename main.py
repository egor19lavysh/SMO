from smo import SMO
import numpy as np
from dataset import Dataset

def test_smo():
    p1 = np.array([[1, 2], [2, 3], [3, 4]])  # Множество P1
    p2 = np.array([[-1, -2], [-2, -3], [-3, -4]])  # Множество P2

    # Объединяем данные в один массив и создаем метки
    X = np.vstack((p1, p2))
    y = np.hstack((np.ones(len(p1)), -np.ones(len(p2))))

    # Создаем и обучаем модель
    smo = SMO()
    smo.fit(X, y)

    # Проверяем предсказания
    predictions = smo.predict(X)
    assert np.all(predictions == y), "Предсказания модели не совпадают с истинными метками!"

    # Проверяем, что гиперплоскость корректно разделяет данные
    w, beta = smo.get_сoefficients()
    for i in range(len(X)):
        decision = np.dot(X[i], w) + beta
        if y[i] == 1:
            assert decision >= 0, "Точка из P1 лежит по ошибочной стороне гиперплоскости!"
        else:
            assert decision < 0, "Точка из P2 лежит по ошибочной стороне гиперплоскости!"

    print("Все тесты пройдены успешно!")

def test_2():
    dt = Dataset()
    dt.create_dataset()
    p1, p2 = dt.get_dataset()
    y = np.hstack((np.ones(25), -np.ones(25)))
    smo = SMO()
    smo.fit(np.vstack((p1[:25], p2[:25])), y)
    print(smo.predict(np.vstack((p2[25:], p1[25:]))))

test_2()