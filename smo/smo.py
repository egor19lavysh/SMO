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
    """

    def __init__(self, C: float = 1.0):
        self._p1 = None
        self._p2 = None
        self._A = None
        self._P = None
        self._m = None
        self._s = None
        self._n = None
        self._w = None
        self._u = None
        self._beta = None
        self.ksi = None
        self._C = C

        self._accuracy = None
        self._precision = None
        self._recall = None
        self._f1_score = None

    def fit(self, X: list[float] | np.ndarray, y: list[float] | np.ndarray) -> None:
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        if len(y) != len(X):
            raise Exception("Длины векторов фичей и меток не совпадают")

        quantile = int(len(X) * 0.25)

        X_test = X[-quantile:]
        y_test = y[-quantile:]

        X = X[:-quantile]
        y = y[:-quantile]

        p1 = X[y == 1]
        p2 = X[y == -1]

        self._p1 = np.array(p1).T
        self._p2 = np.array(p2).T
        self._A = np.hstack((self._p1, -self._p2))
        self._P = np.vstack((p1, p2))
        self._m = self._A.shape[1]
        self._s = self._p1.shape[1]
        self._n = len(p1[0])
        self._ksi = np.array([1 if i < self._s else -1 for i in range(self._m)])
        self._u, self._w = self._train()
        self._beta = self._compute_beta()

        self.evaluate_metrics(X_test, y_test)

    def predict(self, X: list[float] | np.ndarray) -> np.ndarray:
        if isinstance(X, list):
            x = np.array(X)
        if self._w is None or self._beta is None:
            raise Exception("Модель не обучена. Вызовите fit() сначала.")

        decision = np.dot(X, self._w) + self._beta
        return np.where(decision >= 0, 1, -1)

    def _get_indices(self, u) -> tuple[list[int], list[int]]:
        i_streak = list(range(self._s)) + [i for i in range(self._s, self._m) if u[i] > 0]
        i_double_streak = [i for i in range(self._s) if u[i] > 0] + list(range(self._s, self._m))
        return i_streak, i_double_streak

    def _get_i_min(self, v, i_streak) -> int:
        i_min = i_streak[0]
        val_min = np.dot(v, self._P[i_min]) - self._ksi[i_min]
        for i in i_streak[1:]:
            val = np.dot(v, self._P[i]) - self._ksi[i]
            if val < val_min:
                val_min = val
                i_min = i

        return i_min

    def _get_i_max(self, v, i_double_streak) -> int:
        i_max = i_double_streak[0]
        val_max = np.dot(v, self._P[i_max]) - self._ksi[i_max]
        for i in i_double_streak[1:]:
            val = np.dot(v, self._P[i]) - self._ksi[i]
            if val > val_max:
                val_max = val
                i_max = i

        return i_max

    def _compute_delta(self, v, i_min, i_max):
        return np.dot(v, self._P[i_max] - self._P[i_min]) - (self._ksi[i_max] - self._ksi[i_min])

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

    def _check_kkt(self, u, v, tol=1e-3):
        for i in range(self._m):
            grad = np.dot(v, self._P[i]) - self._ksi[i]
            if u[i] == 0 and grad < -tol:
                return False
            elif u[i] == self._C and grad > tol:
                return False
            elif 0 < u[i] < self._C and abs(grad) > tol:
                return False
        return True

    def _compute_beta(self) -> float:
        # Случай 1: Используем опорные векторы (0 < u_i < C)
        mask = (self._u > 0) & (self._u < self._C)
        if np.any(mask):
            beta_values = [
                (1 / self._ksi[i]) - np.dot(self._w, self._P[i])
                for i in np.where(mask)[0]
            ]
            return float(np.median(beta_values))

        # Случай 2: Минимизация T(β)
        beta_grid = np.linspace(-10, 10, 1000)
        losses = [self._compute_T(beta) for beta in beta_grid]
        return float(beta_grid[*np.argmin(losses)])

    def _compute_T(self, beta: float) -> float:
        return sum([np.maximum(0, 1 - self._ksi[i] * (np.dot(self._P, self._w) + beta)) for i in range(self._m)])

    def _train(self, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        s, m, n = self._s, self._m, self._n

        u = np.zeros(m)
        v = np.dot(self._A, u)

        for _ in range(max_iter):

            first_indices, second_indices = self._get_indices(u)

            i_min, i_max = self._get_i_min(v, first_indices), self._get_i_max(v, second_indices)

            delta_k = self._compute_delta(v, i_min=i_min, i_max=i_max)

            if delta_k <= 1e-6:
                break

            lambda_k_hat = delta_k / (np.linalg.norm(self._P[i_min] - self._P[i_max]) ** 2)

            lambda_k = self._compute_lambda_k(i_min, i_max, lambda_k_hat, u)

            u_new = u.copy()

            u_new[i_min] += self._ksi[i_min] * lambda_k
            u_new[i_max] -= self._ksi[i_max] * lambda_k
            v += lambda_k * (self._P[i_min] - self._P[i_max])

            u_new[i_min] = np.clip(u_new[i_min], 0, self._C)
            u_new[i_max] = np.clip(u_new[i_max], 0, self._C)

            current_sum = np.sum(self._ksi * u_new)
            if abs(current_sum) > 1e-6:
                delta = current_sum / (self._ksi[i_min] - self._ksi[i_max])
                u_new[i_min] -= delta * self._ksi[i_min]
                u_new[i_max] += delta * self._ksi[i_max]
                u_new[i_min] = np.clip(u_new[i_min], 0, self._C)
                u_new[i_max] = np.clip(u_new[i_max], 0, self._C)

            v = np.dot(self._A, u_new)
            u = u_new

            if self._check_kkt(u, v):
                break

        return u, v

    def evaluate_metrics(self, X_test: np.ndarray | list, y_test: np.ndarray | list) -> None:
        y_pred = self.predict(X_test)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == -1))
        fn = np.sum((y_pred == -1) & (y_test == 1))
        tn = np.sum((y_pred == -1) & (y_test == -1))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0


        print(tp, fp, tn, fn)
        self._accuracy = float(accuracy)
        self._precision = float(precision)
        self._recall = float(recall)
        self._f1_score = float(f1_score)

    def get_coefficients(self) -> tuple[float, float]:
        return self._w, self._beta

    def get_metrics(self) -> tuple[float, float, float, float]:
        return self._accuracy, self._precision, self._recall, self._f1_score
