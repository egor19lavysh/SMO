import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class Dataset:
    """
    Класс для генерирования n-мерных точек
    """

    def __init__(self, n: int = 2, size: int = 100):
        self._dim = n
        self._size = size
        self._data = None

    def create_explicit_dataset(self, low: int = 0, high: int = 100, alpha: int = 10, with_y: bool = True) -> None:
        mid = (low + high) // 2
        first_half = np.random.randint(low, mid - alpha, (self._size // 2, self._dim))
        second_half = np.random.randint(mid + alpha, high, (self._size // 2, self._dim))
        points = np.vstack((first_half, second_half))

        if with_y:
            labels = np.array([1] * (self._size // 2) + [-1] * (self._size // 2))
            combined = np.hstack((points, labels.reshape(-1, 1)))
            np.random.shuffle(combined)

            points_shuffled = combined[:, :-1]
            labels_shuffled = combined[:, -1]

            data = [points_shuffled, labels_shuffled]
        else:
            data = [points]

        self._data = data

    def create_dataset(self, low: int = 0, high: int = 100, with_y: bool = True) -> None:
        X = np.random.randint(low, high, (self._size, self._dim))
        if with_y:
            y = np.random.choice([-1, 1], self._size)
            data = [X, y]
        else:
            data = [X]

        self._data = data

    def get_dataset(self) -> list[list[int | float]] | np.ndarray:
        return self._data

    def export_to_xlsx(self, filename: str = "dataset.xlsx") -> None:
        if self._data is None:
            raise ValueError("Нет данных для экспорта. Вызовите create_dataset() сначала.")

        if len(self._data) == 2:
            points, labels = self._data
            df = pd.DataFrame(points, columns=[f"x{i+1}" for i in range(self._dim)])
            df["y"] = labels
        else:
            points = self._data[0]
            df = pd.DataFrame(points, columns=[f"x{i + 1}" for i in range(self._dim)])

        df.to_excel(filename, index=False)

    def export_to_csv(self, filename: str = "dataset.csv") -> None:
        if self._data is None:
            raise ValueError("Нет данных для экспорта. Вызовите create_dataset() сначала.")

        if len(self._data) == 2:
            points, labels = self._data
            df = pd.DataFrame(points, columns=[f"x{i + 1}" for i in range(self._dim)])
            df["y"] = labels
        else:
            points = self._data[0]
            df = pd.DataFrame(points, columns=[f"x{i + 1}" for i in range(self._dim)])

        df.to_csv(filename, index=False)