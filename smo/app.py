# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import os
from smo import SMO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Глобальные переменные
X_train = None
y_train = None
model = None
metrics = None  # Для хранения метрик


@app.route('/')
def index():
    return render_template('index.html', image=None, metrics=metrics)


@app.route('/upload_train', methods=['POST'])
def upload_train():
    global X_train, y_train
    if 'file' not in request.files:
        return "Файл не выбран", 400
    file = request.files['file']
    if file.filename == '':
        return "Файл не выбран", 400

    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return "Неподдерживаемый формат файла", 400

        if 'y' not in data.columns:
            return "Столбец 'y' не найден", 400

        X_train = data.drop(columns=['y']).values
        y_train = data['y'].values

        return redirect(url_for('index'))
    except Exception as e:
        return str(e), 400


@app.route('/train', methods=['POST'])
def train():
    global X_train, y_train, model, metrics
    if X_train is None or y_train is None:
        return "Данные не загружены", 400

    try:
        model = SMO(C=1.0)
        model.fit(X_train, y_train)

        image_url = generate_plot(X_train, y_train, model)

        # Получаем метрики
        accuracy, precision, recall, f1_score = model.get_metrics()
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        return render_template('index.html', image=image_url, metrics=metrics)
    except Exception as e:
        return str(e), 400


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return "Модель не обучена", 400

    if 'predict_file' not in request.files:
        return "Файл не выбран", 400

    file = request.files['predict_file']
    if file.filename == '':
        return "Файл не выбран", 400

    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return "Неподдерживаемый формат файла", 400

        X_test = data.values
        y_pred = model.predict(X_test)

        # Сохраняем результаты
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
        pd.DataFrame({'y_pred': y_pred}).to_csv(temp_file, index=False)

        return send_file(temp_file, as_attachment=True, download_name='predictions.csv')
    except Exception as e:
        return str(e), 400


def generate_plot(X, y, model):
    n_features = X.shape[1]
    if n_features not in [2, 3]:
        return None

    fig = plt.figure(figsize=(8, 6))

    if n_features == 2:
        ax = fig.add_subplot(111)
        class_1 = X[y == 1]
        class_2 = X[y == -1]
        ax.scatter(class_1[:, 0], class_1[:, 1], c='blue', label='Класс 1')
        ax.scatter(class_2[:, 0], class_2[:, 1], c='red', label='Класс -1')

        w, beta = model.get_coefficients()
        x_min, x_max = ax.get_xlim()
        y_min = (-beta - w[0] * x_min) / (w[1] if w[1] != 0 else 1e-6)
        y_max = (-beta - w[0] * x_max) / (w[1] if w[1] != 0 else 1e-6)
        ax.plot([x_min, x_max], [y_min, y_max], 'k--', label='Линия разделения')
    else:  # 3D
        ax = fig.add_subplot(111, projection='3d')
        class_1 = X[y == 1]
        class_2 = X[y == -1]
        ax.scatter(class_1[:, 0], class_1[:, 1], class_1[:, 2], c='blue', label='Класс 1')
        ax.scatter(class_2[:, 0], class_2[:, 1], class_2[:, 2], c='red', label='Класс -1')

        w, beta = model.get_coefficients()
        x = np.linspace(*ax.get_xlim(), 10)
        y_vals = np.linspace(*ax.get_ylim(), 10)
        x, y = np.meshgrid(x, y_vals)
        denominator = w[2] if w[2] != 0 else 1e-6
        z = (-w[0] * x - w[1] * y - beta) / denominator
        ax.plot_surface(x, y, z, alpha=0.5, color='green', label='Плоскость разделения')

    ax.legend()
    ax.set_title('График разделения точек')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    if n_features == 3:
        ax.set_zlabel('X3')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return 'data:image/png;base64,' + data


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)