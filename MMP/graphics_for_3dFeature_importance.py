import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # для работы с 3D-графиками
from datetime import datetime

def plot_3d_feature(df, feature, plot_dir):
    """
    Строит 3D-график для выбранного признака.

    По оси X: future_window
    По оси Y: rolling_window
    По оси Z: значение признака (например, TrendSELL)

    График сохраняется в папке plot_dir.
    """
    # Собираем сводную таблицу: строки – rolling_window, столбцы – future_window,
    # значения – значение признака.
    try:
        pivot = df.pivot(index='rolling_window', columns='future_window', values=feature)
    except Exception as e:
        print(f"Ошибка при построении pivot таблицы для {feature}: {e}")
        return

    # Создаем сетку для осей X и Y
    X = pivot.columns.values
    Y = pivot.index.values
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z = pivot.values

    # Построение 3D-графика
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('future_window')
    ax.set_ylabel('rolling_window')
    ax.set_zlabel(feature)
    ax.set_title(f'3D-график: {feature} от future_window и rolling_window')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()

    # Сохраняем график в файл
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"3d_{feature}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График '3d_{feature}' сохранён: {filename}")

# Чтение файла log.txt и сбор данных по экспериментам
experiments = []
with open('log.txt', 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file.readlines()]
    current_exp = {}
    for line in lines:
        # Если встречается строка с "Запуск эксперимента" – начинаем новый эксперимент
        if "Запуск эксперимента" in line:
            if current_exp:
                experiments.append(current_exp)
            current_exp = {}
            s1 = line.find("future_window")
            s3 = line.find(";")
            s2 = line.find("rolling_window")
            current_exp["future_window"] = int(line[s1 + 14: s3])
            current_exp["rolling_window"] = int(line[s2 + 15:])
        # Для каждой строки с признаком добавляем данные в текущий эксперимент
        if "TrendSELL" in line:
            s1 = line.find("0")
            current_exp["TrendSELL"] = float(line[s1 - 1:])
        if "BondSELL" in line:
            s1 = line.find("0")
            current_exp["BondSELL"] = float(line[s1 - 1:])
        if "DiffFromTrendSELL" in line:
            s1 = line.find("0")
            current_exp["DiffFromTrendSELL"] = float(line[s1 - 1:])
        if "MomentumSELL" in line:
            s1 = line.find("0")
            current_exp["MomentumSELL"] = float(line[s1 - 1:])
        if "RollingSTD_SELL" in line:
            s1 = line.find("0")
            current_exp["RollingSTD_SELL"] = float(line[s1 - 1:])
        if "VolumeSELL" in line:
            s1 = line.find("0")
            current_exp["VolumeSELL"] = float(line[s1 - 1:])
    # Добавляем последний собранный эксперимент, если он не пустой
    if current_exp:
        experiments.append(current_exp)

for exp in experiments:
    print(exp)
print("Количество экспериментов:", len(experiments))

# Преобразуем список экспериментов в DataFrame
df_experiments = pd.DataFrame(experiments)
print(df_experiments.head())

# Указываем папку для сохранения графиков
plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

# Построим 3D-график для выбранного признака, например, TrendSELL
plot_3d_feature(df_experiments, "TrendSELL", plot_dir)

# Если необходимо построить графики для всех признаков, используем цикл:
features = ["TrendSELL", "BondSELL", "DiffFromTrendSELL", "MomentumSELL", "RollingSTD_SELL", "VolumeSELL"]
for feature in features:
    plot_3d_feature(df_experiments, feature, plot_dir)
