import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # для работы с 3D-графиками
from datetime import datetime


def plot_3d_metric(df, metric, plot_dir):
    """
    Строит 3D-график для выбранной метрики, где:
      OX: future_window
      OY: rolling_window
      OZ: значение метрики (например, "Количество совершённых сделок" или "Итоговый баланс")

    График сохраняется в папке plot_dir.
    """
    try:
        pivot = df.pivot(index='rolling_window', columns='future_window', values=metric)
    except Exception as e:
        print(f"Ошибка при построении pivot таблицы для {metric}: {e}")
        return

    X = pivot.columns.values  # future_window
    Y = pivot.index.values  # rolling_window
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z = pivot.values  # значения метрики

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('future_window')
    ax.set_ylabel('rolling_window')
    ax.set_zlabel(metric)
    ax.set_title(f'3D-график: {metric} от future_window и rolling_window')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"3d_{metric.replace(' ', '_')}_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График '{metric}' сохранён: {filename}")


# Предположим, что у вас уже есть DataFrame, сформированный из лог-файла:
experiments = []
with open('log.txt', 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file.readlines()]
    current_exp = {}
    for line in lines:
        # Если встречается строка с "Запуск эксперимента" – начинаем новый эксперимент.
        if "Запуск эксперимента" in line:
            if current_exp:
                experiments.append(current_exp)
            current_exp = {}
            s1 = line.find("future_window")
            s3 = line.find(";")
            s2 = line.find("rolling_window")
            current_exp["future_window"] = int(line[s1 + 14: s3])
            current_exp["rolling_window"] = int(line[s2 + 15:])
        if "Итоговый баланс" in line:
            s1 = line.find(":")
            # Используем регулярное выражение или срез для извлечения числа; здесь предполагается, что после двоеточия идет число
            try:
                # Убираем возможные лишние символы, если они есть
                num_str = line[s1 + 2:].split()[0]
                current_exp["Итоговый баланс"] = float(num_str)
            except Exception as e:
                print("Ошибка при извлечении Итогового баланса:", e)
        if "Количество совершённых сделок" in line:
            s1 = line.find(":")
            try:
                num_str = line[s1 + 2:].split()[0]
                current_exp["Количество совершённых сделок"] = float(num_str)
            except Exception as e:
                print("Ошибка при извлечении Количества сделок:", e)
    if current_exp:
        experiments.append(current_exp)

for exp in experiments:
    print(exp)
print("Количество экспериментов:", len(experiments))

# Преобразуем список экспериментов в DataFrame.
df_experiments = pd.DataFrame(experiments)
print("Собранные данные (первые 5 строк):")
print(df_experiments.head())

# Указываем папку для сохранения графиков
plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

# Построим 3D-график для "Количество совершённых сделок"
plot_3d_metric(df_experiments, "Количество совершённых сделок", plot_dir)

# Построим 3D-график для "Итоговый баланс"
plot_3d_metric(df_experiments, "Итоговый баланс", plot_dir)
