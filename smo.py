import numpy as np


class SMO:
    """
    Класс для реализации алгоритма SMO.
    Принимает на вход два множества точек; при этом гарантируется,
    что выпуклые оболочки множеств не пересекаются.
    Класс строит гиперплоскость для отделения точек двух множеств.

    Parameters
    ----------
    p1 : list of int or nd.array
        Множество точек P1
    p2 : list of int or nd.array
        Множество точек P2

    Returns
    -------
    None
    """

    def __init__(self):
        self._p1 = None
        self._p2 = None
        self._a = None
        self._p = None
        self._m = None
        self._s = None
        self._n = None
        self._w = None
        self._beta = None

    def fit(self, x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> None:
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        if len(y) != len(x):
            raise Exception("Длины векторов фичей и меток не совпадают")

        p1 = x[y == 1]
        p2 = x[y == -1]

        self._p1 = np.array(p1).T
        self._p2 = np.array(p2).T
        self._a = np.hstack((self._p1, -self._p2))
        self._p = np.vstack((p1, p2))
        self._m = self._a.shape[1]
        self._s = self._p1.shape[1]
        self._n = len(p1[0])

        self._w = self._train()
        self._beta = self._compute_beta()

    def predict(self, x: list[float] | np.ndarray) -> np.ndarray:
        if isinstance(x, list):
            x = np.array(x)
        if self._w is None or self._beta is None:
            raise Exception("Модель не обучена. Вызовите fit() сначала.")

        decision = np.dot(x, self._w) + self._beta
        return np.where(decision >= 0, 1, -1)

    def _ksi(self, i: int) -> int:
        return 1 if i < self._s else -1

    def _get_indices(self, u) -> tuple[list[int], list[int]]:
        i_streak = list(range(self._s)) + [i for i in range(self._s, self._m) if u[i] > 0]
        i_double_streak = [i for i in range(self._s) if u[i] > 0] + list(range(self._s, self._m))
        return i_streak, i_double_streak

    def _get_i_min(self, v, i_streak) -> int:
        i_min = i_streak[0]
        val_min = np.dot(v, self._p[i_min]) - self._ksi(i_min)
        for i in i_streak[1:]:
            val = np.dot(v, self._p[i]) - self._ksi(i)
            if val < val_min:
                val_min = val
                i_min = i

        return i_min

    def _get_i_max(self, v, i_double_streak) -> int:
        i_max = i_double_streak[0]
        val_max = np.dot(v, self._p[i_max]) - self._ksi(i_max)
        for i in i_double_streak[1:]:
            val = np.dot(v, self._p[i]) - self._ksi(i)
            if val > val_max:
                val_max = val
                i_max = i

        return i_max

    def _compute_delta(self, v, i_min, i_max):
        return np.dot(v, self._p[i_max] - self._p[i_min]) - (self._ksi(i_max) - self._ksi(i_min))

    def _compute_lambda_k(self, i_min: int, i_max: int, lambda_k_hat: float, u) -> float:
        s, m, n = self._s, self._m, self._n
        if s <= i_min <= m and s <= i_max <= m:
            lambda_k = min(lambda_k_hat, u[i_min])
        elif i_min < s and i_max < s:
            lambda_k = min(lambda_k_hat, u[i_max])
        elif m >= i_min >= s > i_max:
            lambda_k = min(lambda_k_hat, u[i_max], u[i_min])
        else:
            lambda_k = lambda_k_hat
        return lambda_k

    def _compute_beta(self) -> float:
        low = max([float(1 - np.dot(self._w, p_i)) for p_i in self._p1.T])
        high = min([float(-1 - np.dot(self._w, p_i)) for p_i in self._p2.T])
        beta = (low + high) / 2
        return beta

    def _train(self, max_iter: int = 1000) -> np.ndarray:
        s, m, n = self._s, self._m, self._n

        u = np.zeros(m)
        v = np.dot(self._a, u)

        for _ in range(max_iter):

            first_indices, second_indices = self._get_indices(u)

            i_min, i_max = self._get_i_min(v, first_indices), self._get_i_max(v, second_indices)

            delta_k = self._compute_delta(v, i_min=i_min, i_max=i_max)

            if delta_k == 0:
                return v

            lambda_k_hat = delta_k / (np.linalg.norm(self._p[i_min] - self._p[i_max]) ** 2)

            lambda_k = self._compute_lambda_k(i_min, i_max, lambda_k_hat, u)

            u_new = u.copy()
            u_new[i_min] += self._ksi(i_min) * lambda_k
            u_new[i_max] -= self._ksi(i_max) * lambda_k
            v += lambda_k * (self._p[i_min] - self._p[i_max])

            u = u_new

        return v

    def get_сoefficients(self) -> tuple[float, float]:
        return self._w, self._beta
