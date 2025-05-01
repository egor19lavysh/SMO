import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from smo import SMO


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SMO Classifier")


        self.X_train = None
        self.y_train = None
        self.model = None


        self.create_widgets()

    def create_widgets(self):

        self.label_load = tk.Label(self.root, text="Загрузите данные (CSV или XLSX):")
        self.label_load.pack(pady=10)

        self.button_load = tk.Button(self.root, text="Загрузить", command=self.load_data)
        self.button_load.pack(pady=5)

        self.button_train = tk.Button(self.root, text="Обучить модель", command=self.train_model, state=tk.DISABLED)
        self.button_train.pack(pady=5)

        self.button_predict = tk.Button(
            self.root, text="Предсказать", command=self.predict_data, state=tk.DISABLED
        )
        self.button_predict.pack(pady=5)

        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")


            if "y" not in data.columns:
                raise ValueError("Столбец 'y' не найден в данных")


            self.X_train = data.drop(columns=["y"]).values
            self.y_train = data["y"].values


            self.button_train.config(state=tk.NORMAL)
            messagebox.showinfo("Успех", "Данные успешно загружены!")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Ошибка", "Данные не загружены")
            return

        try:

            self.model = SMO(C=1.0)
            self.model.fit(self.X_train, self.y_train)

            self.button_predict.config(state=tk.NORMAL)
            messagebox.showinfo("Успех", "Модель успешно обучена!")


            if self.X_train.shape[1] == 2:
                self.plot_data_and_decision_boundary()

            self.plot_data_and_decision_boundary()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def predict_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла")

            X_test = data.values

            if self.model is None:
                raise ValueError("Модель не обучена")

            y_pred = self.model.predict(X_test)

            output_path = filedialog.asksaveasfilename(defaultextension=".csv")
            if output_path[-4:] == ".csv":
                pd.DataFrame({"y_pred": y_pred}).to_csv(output_path, index=False)
                messagebox.showinfo("Успех", f"Результаты сохранены в {output_path}")
            elif output_path[-4:] == "xlsx":
                pd.DataFrame({"y_pred": y_pred}).to_excel(output_path, index=False)
                messagebox.showinfo("Успех", f"Результаты сохранены в {output_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def plot_data_and_decision_boundary(self):
        if self.X_train is None or self.y_train is None or self.model is None:
            return

        n_features = self.X_train.shape[1]
        if n_features > 3:
            raise ValueError("Визуализация поддерживается только для данных размерности 2 или 3.")


        self.ax.clear()

        if n_features == 3:
            if not hasattr(self.ax, 'zaxis'):
                self.ax.remove()
                self.ax = self.figure.add_subplot(111, projection='3d')
        else:
            if hasattr(self.ax, 'zaxis'):
                self.ax.remove()
                self.ax = self.figure.add_subplot(111)

        class_1 = self.X_train[self.y_train == 1]
        class_2 = self.X_train[self.y_train == -1]

        if n_features == 2:
            self.ax.scatter(class_1[:, 0], class_1[:, 1], c="blue", label="Класс 1")
            self.ax.scatter(class_2[:, 0], class_2[:, 1], c="red", label="Класс -1")
        elif n_features == 3:
            self.ax.scatter(
                class_1[:, 0], class_1[:, 1], class_1[:, 2],
                c="blue", label="Класс 1"
            )
            self.ax.scatter(
                class_2[:, 0], class_2[:, 1], class_2[:, 2],
                c="red", label="Класс -1"
            )

        w, beta = self.model.get_coefficients()
        if n_features == 2:
            x_min, x_max = self.ax.get_xlim()
            y_min = (-beta - w[0] * x_min) / (w[1] if w[1] != 0 else 1e-6)
            y_max = (-beta - w[0] * x_max) / (w[1] if w[1] != 0 else 1e-6)
            self.ax.plot([x_min, x_max], [y_min, y_max], "k--", label="Линия разделения")
        elif n_features == 3:
            x = np.linspace(*self.ax.get_xlim(), 10)
            y = np.linspace(*self.ax.get_ylim(), 10)
            x, y = np.meshgrid(x, y)
            denominator = w[2] if w[2] != 0 else 1e-6
            z = (-w[0] * x - w[1] * y - beta) / denominator
            self.ax.plot_surface(x, y, z, alpha=0.5, color='green', label="Плоскость разделения")

        self.ax.legend()
        self.ax.set_title("График разделения точек")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        if n_features == 3:
            self.ax.set_zlabel("X3")
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
