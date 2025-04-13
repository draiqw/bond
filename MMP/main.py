import os

from sklearn.linear_model import LinearRegression

from src.loadFile import make_df_all
from src.train import prepare_dataset_split, train_model_split
from src.simulate_trading import simulate_trading
from src.graphics import (
    collect_and_plot_3d,
    plot_bond_prices_over_time,
    plot_deviations_from_trend,
    plot_threshold_sensitivity,
    plot_3d_balance,
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_equity_curve,
    plot_trading_positions
)


plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

def run_experiment(future_window, rolling_window):
    df_all = make_df_all()
    df_all = df_all.drop(['futures', 'stocks'], axis=1)

    df_train, df_test = prepare_dataset_split(
        df_all,
        future_window,
        rolling_window
    )

    df_test, model_sell, importance = train_model_split(
        df_train,
        df_test,
        model=LinearRegression,
        need_training=True
    )

    plot_feature_importance(importance, plot_dir)

    final_balance, trades = simulate_trading(
        df_test,
        future_window=future_window,
        threshold=0.1,
        commission_rate=0.23 / 2,
        initial_balance=100000
    )

    plot_predicted_vs_actual(df_test, plot_dir)
    plot_equity_curve(trades, plot_dir)
    plot_threshold_sensitivity(df_test, plot_dir)
    plot_trading_positions(df_test, trades, plot_dir)

    print("Итоговый баланс:", final_balance)
    print("Количество совершённых сделок:", len(trades))


    return df_test, final_balance, trades




run_experiment(future_window=1500, rolling_window=200)

future_windows = [100, 500, 1000, 1500, 2000, 2500, 3000]
rolling_windows = [10, 20, 50, 100, 200, 500, 1000, 1500]
plot_3d_balance(future_windows, rolling_windows, run_experiment, plot_dir)

