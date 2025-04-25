import numpy as np
from matplotlib import pyplot as plt


class Dataset:
    """
    Класс для генерирования n-мерных точек
    """

    def __init__(self, n: int = 2, size: int = 100):
        self._dim = n
        self._size = size
        self._data = None

    def create_dataset(self, low: int = 0, high: int = 100, alpha: int = 10) -> None:
        mid = (low + high) // 2
        first_half = np.random.randint(low, mid - alpha, (self._size // 2, self._dim))
        second_half = np.random.randint(mid + alpha, high, (self._size // 2, self._dim))
        data = [first_half, second_half]

        self._data = data

    def get_dataset(self) -> list[list[int | float]] | np.ndarray:
        return self._data

    def plot_dataset(self) -> None:
        data = np.array(np.vstack(self._data))
        if self._dim == 2:
            x = data[:, 0]
            y = data[:, 1]
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="red")
            plt.title("Исходный датасет")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
        elif self._dim == 3:
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z, color="red")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        else:
            raise Exception("График может быть построен только при размерности 2 или 3")
