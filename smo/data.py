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
    w, beta = smo.get_coefficients()
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


import pandas as pd


def test_3():
    dt = Dataset(n=3)
    dt.create_dataset()
    p1, p2 = dt.get_dataset()

    # Проверяем размерности
    print(f"p1 shape: {p1.shape}, p2 shape: {p2.shape}")  # Для отладки

    # Создаем метки: 25 меток "1" для p1 и 25 меток "-1" для p2
    y = np.hstack((np.ones(p1.shape[0])[:25], -np.ones(p2.shape[0])[:25]))

    # Объединяем данные и метки
    data = np.vstack((p1[:25], p2[:25]))
    data_2 = np.vstack((p1[25:], p2[25:])) # размер (50, n)
    labels = y.reshape(-1, 1)  # размер (50, 1)

    # Проверяем совпадение размерностей
    assert data.shape[0] == labels.shape[0], "Несовпадение размеров данных и меток!"

    columns = [f"Feature_{i}" for i in range(data.shape[1])] + ["y"]
    df = pd.DataFrame(
        np.hstack((data, labels)),
        columns=columns
    )

    df2 = pd.DataFrame(
        data_2
    )

    # Сохраняем в XLSX
    with pd.ExcelWriter("dataset.xlsx") as writer:
        df.to_excel(writer, index=False)

    with pd.ExcelWriter("dataset2.xlsx") as writer:
        df2.to_excel(writer, index=False)

    # Остальная логика...

test_3()