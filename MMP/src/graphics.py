import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.simulate_trading import simulate_trading
from datetime import datetime


def collect_and_plot_3d(
        future_windows,
        rolling_windows,
        run_experiment,
        plot_dir
):
    results = []

    for fw in future_windows:
        for rw in rolling_windows:
            print(
                f"\nЗапуск: future_window={fw};"
                f"rolling_window={rw}\n"
            )

            _, _, final_balance, trades = run_experiment(
                future_window=fw,
                rolling_window=rw
            )

            print(f"Итоговый баланс: {final_balance}")
            print(f"Количество сделок: {len(trades)}")

            results.append({
                'future_window': fw,
                'rolling_window': rw,
                'balance': final_balance
            })

    df_results = pd.DataFrame(results)
    print("\nИтоговая таблица:\n", df_results)

    pivoted = df_results.pivot(
        index='rolling_window',
        columns='future_window',
        values='balance'
    )

    X = pivoted.columns.values
    Y = pivoted.index.values
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z_mesh = pivoted.values

    # Теперь строим 3D-график
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X_mesh,
        Y_mesh,
        Z_mesh,
        cmap='viridis',
        edgecolor='none'
    )

    ax.set_xlabel('Future Window')
    ax.set_ylabel('Rolling Window')
    ax.set_zlabel('Final Balance')
    ax.set_title('3D-зависимость баланса от future_window и rolling_window')

    ax.grid(True)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(
        plot_dir,
        f"3D_balance_plot_{timestamp}.png"
    )
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n3D-график сохранён в: {filename}")


# ==============================
# 1. Цены синтетических облигаций во времени с возможностью выбора среза времени
# ==============================
def plot_bond_prices_over_time(df, plot_dir, start_time=None, end_time=None):
    """
    Строит график цен облигаций (BondBUY и BondSELL) во времени.
    По оси X: time_dt
    По оси Y: цена облигации (%).

    Параметры:
      - start_time, end_time (опционально): фильтр по времени (строка или datetime).
    График сохраняется в папке plot_dir.
    """
    df_plot = df.copy()
    # Если указан временной срез, фильтруем DataFrame
    if start_time is not None and end_time is not None:
        df_plot = df_plot[(df_plot['time_dt'] >= pd.to_datetime(start_time)) &
                          (df_plot['time_dt'] <= pd.to_datetime(end_time))]

    plt.figure(figsize=(10, 5))
    plt.plot(df_plot['time_dt'], df_plot['BondBUY'], label='BondBUY', linewidth=1.5)
    plt.plot(df_plot['time_dt'], df_plot['BondSELL'], label='BondSELL', linewidth=1.5)
    plt.xlabel('Время')
    plt.ylabel('Цена облигации (%)')
    plt.title('Цены синтетических облигаций во времени')
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"bond_prices_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'bond_prices' сохранён: {filename}")


# ==============================
# 2. Отклонения от тренда
# ==============================
def plot_deviations_from_trend(df, plot_dir):
    """
    Строит два графика:
    - Линейный график: BondSELL и TrendSELL по времени.
    - Гистограмма DiffFromTrendSELL.

    График сохраняется в папке plot_dir.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Линейный график
    ax1.plot(df['time_dt'], df['BondSELL'], label='BondSELL', linewidth=1.5)
    ax1.plot(df['time_dt'], df['TrendSELL'], label='TrendSELL', linewidth=1.5)
    ax1.set_ylabel('Цена (%)')
    ax1.set_title('BondSELL и TrendSELL во времени')
    ax1.legend()
    ax1.grid(True)

    # Гистограмма отклонений
    ax2.hist(df['DiffFromTrendSELL'], bins=50, color='skyblue', edgecolor='black')
    ax2.set_xlabel('DiffFromTrendSELL')
    ax2.set_ylabel('Частота')
    ax2.set_title('Гистограмма отклонений от тренда')
    ax2.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"deviations_from_trend_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'deviations_from_trend' сохранён: {filename}")


# ==============================
# 3. Пороговая сетка: прибыль и метрики
# ==============================
def plot_threshold_sensitivity(
        df,
        plot_dir,
        future_window=1500,
        commission_rate=0.23 / 2,
        initial_balance=100000.0,
        thresholds=np.arange(0.01, 0.31, 0.01)
):
    """
    Для каждого порога threshold от 0.01 до 0.3:
      - рассчитывается итоговый баланс и количество сделок;
      - вычисляется средняя прибыль на сделку.

    Строятся 3 графика:
      1. Threshold → Прибыль (итоговый баланс - начальный).
      2. Threshold → Количество сделок.
      3. Threshold → Средняя прибыль на сделку.

    График сохраняется в папке plot_dir.
    """
    profits = []
    trades_counts = []
    avg_profit_per_trade = []

    for thr in thresholds:
        final_balance, trade_logs = simulate_trading(
            df.copy(),
            threshold=thr,
            commission_rate=commission_rate,
            initial_balance=initial_balance,
            future_window=future_window
        )
        profit = final_balance - initial_balance
        profits.append(profit)
        count_trades = sum(1 for trade in trade_logs if trade['action'] == 'sell')
        trades_counts.append(count_trades)
        avg_profit = profit / count_trades if count_trades > 0 else np.nan
        avg_profit_per_trade.append(avg_profit)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(thresholds, profits, marker='o', linewidth=1.5)
    axs[0].set_ylabel('Прибыль (руб.)')
    axs[0].set_title('Зависимость прибыли от threshold')
    axs[0].grid(True)

    axs[1].plot(thresholds, trades_counts, marker='o', linewidth=1.5, color='orange')
    axs[1].set_ylabel('Кол-во сделок')
    axs[1].set_title('Зависимость кол-ва сделок от threshold')
    axs[1].grid(True)

    axs[2].plot(thresholds, avg_profit_per_trade, marker='o', linewidth=1.5, color='green')
    axs[2].set_xlabel('Threshold')
    axs[2].set_ylabel('Средняя прибыль на сделку (руб.)')
    axs[2].set_title('Средняя прибыль на сделку')
    axs[2].grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"threshold_sensitivity_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'threshold_sensitivity' сохранён: {filename}")


# ==============================
# 4. 3D-график зависимости прибыли от future_window и rolling_window
# ==============================
def plot_3d_balance(
        future_windows,
        rolling_windows,
        run_experiment,
        plot_dir
):
    """
    Для заданных диапазонов future_window и rolling_window запускает эксперимент и строит 3D-график,
    где по оси X – future_window, по Y – rolling_window, по Z – итоговый баланс.

    График сохраняется в папке plot_dir.
    """
    results = []
    for fw in future_windows:
        for rw in rolling_windows:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Запуск эксперимента: future_window={fw}; rolling_window={rw}")
            _, final_balance, trade_logs = run_experiment(future_window=fw, rolling_window=rw)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Итоговый баланс: {final_balance}, Кол-во сделок: {len(trade_logs)}")
            results.append({
                'future_window': fw,
                'rolling_window': rw,
                'balance': final_balance
            })

    df_results = pd.DataFrame(results)
    pivoted = df_results.pivot(index='rolling_window', columns='future_window', values='balance')
    X = pivoted.columns.values
    Y = pivoted.index.values
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z_mesh = pivoted.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Future Window')
    ax.set_ylabel('Rolling Window')
    ax.set_zlabel('Final Balance')
    ax.set_title('3D-зависимость баланса от future_window и rolling_window')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"3D_balance_plot_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 3D-график сохранён: {filename}")


# ==============================
# 5. Feature importance (горизонтальная диаграмма)
# ==============================
def plot_feature_importance(importance_series, plot_dir):
    """
    Строит горизонтальный barh-график важности фичей.
    Важности сортируются по модулю.

    График сохраняется в папке plot_dir.
    """
    importance_sorted = importance_series.reindex(importance_series.abs().sort_values(ascending=True).index)
    plt.figure(figsize=(8, 4))
    plt.barh(importance_sorted.index, importance_sorted.values, color='mediumblue')
    plt.xlabel('Коэффициент')
    plt.title('Feature Importance (сортировка по модулю)')
    plt.grid(True, axis='x')
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"feature_importance_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'feature_importance' сохранён: {filename}")


# ==============================
# 6. Предсказанная vs настоящая цена через n шагов
# ==============================
def plot_predicted_vs_actual(df, plot_dir):
    """
    Строит line plot для сравнения target_sell (настоящая цена)
    и PredSELL (предсказанная цена).
    По оси X – индекс, по оси Y – значение цены.

    График сохраняется в папке plot_dir.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['target_sell'], label='TargetSELL', linewidth=1.5)
    plt.plot(df.index, df['PredSELL'], label='PredSELL', linewidth=1.5)
    plt.xlabel('Индекс')
    plt.ylabel('Цена (%)')
    plt.title('Сравнение предсказанной и настоящей цены (n-шагов)')
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"predicted_vs_actual_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'predicted_vs_actual' сохранён: {filename}")


# ==============================
# 7. График роста баланса (equity curve)
# ==============================
def plot_equity_curve(trade_logs, plot_dir):
    """
    Строит график equity curve на основе trade_logs.
    По оси X – tick, по оси Y – баланс.

    График сохраняется в папке plot_dir.
    """
    ticks = [trade['tick'] for trade in trade_logs]
    balances = [trade['balance'] for trade in trade_logs]

    plt.figure(figsize=(10, 5))
    plt.plot(ticks, balances, marker='o', linewidth=1.5)
    plt.xlabel('Tick')
    plt.ylabel('Баланс (руб.)')
    plt.title('График роста баланса (Equity Curve)')
    plt.grid(True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"equity_curve_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'equity_curve' сохранён: {filename}")


# ==============================
# 8. График сделок на фоне цены
# ==============================
def plot_trading_positions(df, trade_logs, plot_dir):
    """
    Строит график цен (BondSELL и TrendSELL) с нанесёнными точками:
      - Зелёными точками: открытие позиции (action == 'start_short')
      - Красными точками: закрытие позиции (action == 'sell' или 'sell_partial')
    Используется столбец time_dt для оси X.

    График сохраняется в папке plot_dir.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df['time_dt'], df['BondSELL'], label='BondSELL', linewidth=1.5)
    plt.plot(df['time_dt'], df['TrendSELL'], label='TrendSELL', linewidth=1.5)

    open_times, open_prices = [], []
    close_times, close_prices = [], []

    for trade in trade_logs:
        tick = trade['tick']
        if tick < len(df):
            time_val = df.iloc[tick]['time_dt']
            price_val = df.iloc[tick]['BondSELL']
            if trade['action'] == 'start_short':
                open_times.append(time_val)
                open_prices.append(price_val)
            elif trade['action'] in ['sell', 'sell_partial']:
                close_times.append(time_val)
                close_prices.append(price_val)

    plt.scatter(open_times, open_prices, color='green', marker='o', label='Открытие позиции')
    plt.scatter(close_times, close_prices, color='red', marker='o', label='Закрытие позиции')
    plt.xlabel('Время')
    plt.ylabel('Цена (%)')
    plt.title('Сделки на фоне цен BondSELL / TrendSELL')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(plot_dir, f"trading_positions_{timestamp}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] График 'trading_positions' сохранён: {filename}")
