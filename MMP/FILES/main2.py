import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import joblib

from src.loadFile import make_df_all
from src.graphics import (
    plot_bond_buy,
    plot_bond_sell,
    clear_directory,
    plot_bond_prices
)
from src.prints import (
    print1,
    print2,
    print3,
    print4,
    print_metrics
)

plot_dir = './plots'

def generate_features(
        df,
        window,
        diff
):
    df['TrendBUY'] = df['BondBUY'].rolling(window=window, closed='left').mean()
    df['TrendSELL'] = df['BondSELL'].rolling(window=window, closed='left').mean()

    df['MomentumBUY'] = df['BondBUY'].diff(diff)
    df['RollingSTD_BUY'] = df['BondBUY'].rolling(window=window, closed='left').std()
    df['DiffFromTrendBUY'] = df['BondBUY'] - df['TrendBUY']

    df['MomentumSELL'] = df['BondSELL'].diff(diff)
    df['RollingSTD_SELL'] = df['BondSELL'].rolling(window=window, closed='left').std()
    df['DiffFromTrendSELL'] = df['BondSELL'] - df['TrendSELL']

    return df.dropna()

def calculate_bond_prices(df):
    df['BondBUY'] = (df['BID_F_P0'] / df['OFFER_S_P0'] - 1) * 100
    df['BondSELL'] = (df['OFFER_F_P0'] / df['BID_S_P0'] - 1) * 100
    df['VolumeBUY'] = df[['BID_F_Q0', 'OFFER_S_Q0']].min(axis=1)
    df['VolumeSELL'] = df[['OFFER_F_Q0', 'BID_S_Q0']].min(axis=1)

    return df.dropna()

def generate_target(
        df,
        future_window
):
    df['target_buy'] = df['BondBUY'].shift(-future_window)
    df['target_sell'] = df['BondSELL'].shift(-future_window)

    return df.dropna()

def train_model_split(
        df_train,
        df_test,
        model,
        need_training:bool,
):
    def make_features(label):
        return [
            'Bond' + label,
            'Trend' + label,
            'Volume' + label,
            'Momentum' + label,
            'RollingSTD_' + label,
            'DiffFromTrend' + label
        ]

    if need_training:
        model_buy = model()
        model_sell = model()

        model_buy.fit(df_train[make_features('BUY')], df_train['target_buy'])

        model_sell.fit(df_train[make_features('SELL')], df_train['target_sell'])

        df_test['PredBUY'] = model_buy.predict(df_test[make_features('BUY')])
        df_test['PredSELL'] = model_sell.predict(df_test[make_features('SELL')])

        joblib.dump(model_buy, './models/model_buy.pkl')
        joblib.dump(model_sell, './models/model_sell.pkl')

    else:
        model_buy = joblib.load('./models/model_buy.pkl')
        model_sell = joblib.load('./models/model_sell.pkl')

        df_test['PredBUY'] = model_buy.predict(df_test[make_features('BUY')])
        df_test['PredSELL'] = model_sell.predict(df_test[make_features('SELL')])

    print_metrics(df_test['target_buy'], df_test['PredBUY'], "bond_buy")
    print_metrics(df_test['target_sell'], df_test['PredSELL'], "bond_sell")
    return df_test, model_buy, model_sell

def prepare_dataset_split(
        df_all,
        future_window,
        rolling_window
):
    df_train = df_all[df_all['week'].isin([0, 1])].copy()
    df_test = df_all[df_all['week'].isin([2, 3, 4])].copy()

    for name, df in [('train', df_train), ('test', df_test)]:
        df = calculate_bond_prices(df)
        df = generate_features(df, window=rolling_window, diff=5)
        df = generate_target(df, future_window=future_window)

        if name == 'train':
            df_train = df
        else:
            df_test = df

    return df_train, df_test

def simulate_trading(
        df,
        threshold: float,
        initial_balance=100000.0,
        commission_rate=0.23 / 2,
        future_window=5,
):
    balance = initial_balance
    position = None
    trade_logs = []

    for tick in range(len(df)):
        row = df.iloc[tick]

        bond_buy = row['BondBUY']
        bond_sell = row['BondSELL']

        offer_stock = row['OFFER_S_P0']
        bid_stock = row['BID_S_P0']

        volume_buy = row['VolumeBUY']
        volume_sell = row['VolumeSELL']

        bond_buy_rub = bond_buy * offer_stock / 100
        bond_sell_rub = bond_sell * bid_stock / 100

        pred_sell = row['PredSELL']

        if (bond_buy <= 0) or (bond_sell <= 0):
            continue

        # print(f""
        #       f"Tick={tick}, "
        #       f"bond_buy={bond_buy:.2f},"
        #       f"bond_sell={bond_sell:.2f},"
        #       f"pred_sell={pred_sell:.2f},"
        #       f"offer_stock={offer_stock:.2f},"
        #       f"bid_stock={bid_stock:.2f},"
        #       f"bond_buy_rub={bond_buy_rub:.2f},"
        #       f"bond_sell_rub={bond_sell_rub:.2f}"
        #
        # )

        if position is None:
            if (bond_buy - pred_sell) > threshold:
                cand_volume = int(balance // offer_stock)
                volume = min(volume_buy, cand_volume)

                if volume > 0:
                    cost = bond_buy * volume
                    commission_buy = volume * commission_rate
                    total_cost = cost + commission_buy

                    if balance >= total_cost:
                        balance -= total_cost

                        position = {
                            'entry_price': bond_buy_rub,
                            'volume': volume,
                            'entry_tick': tick,
                            'commission_buy': commission_buy
                        }
                        trade_logs.append({
                            'action': 'start_short',
                            'tick': tick,
                            'price_perc': bond_buy,
                            'price_rub': bond_buy_rub,
                            'volume': volume,
                            'commission': commission_buy,
                            'balance': balance
                        })
        else:
            ticks_in_position = tick - position['entry_tick']
            if ticks_in_position >= future_window:

                vol_in_pos = position['volume']

                if volume_sell >= vol_in_pos:

                    exit_price = bond_sell

                    proceeds = exit_price * vol_in_pos
                    commission_sell = vol_in_pos * commission_rate
                    balance += proceeds - commission_sell

                    trade_logs.append({
                        'action': 'sell',
                        'tick': tick,
                        'price': bond_sell,
                        'volume': vol_in_pos,
                        'commission': commission_sell,
                        'balance': balance
                    })

                    position = None
                else:
                    exit_price_rub = bond_sell_rub

                    partial_volume = volume_sell

                    if partial_volume > 0:
                        proceeds = exit_price_rub * partial_volume
                        commission_sell = partial_volume * commission_rate
                        balance += proceeds - commission_sell

                        close_cost = exit_price_rub * partial_volume + commission_sell

                        trade_logs.append({
                            'action': 'sell_partial',
                            'tick': tick,
                            'price_perc': bond_sell,
                            'price_rub': exit_price_rub,
                            'volume': partial_volume,
                            'commission': commission_sell,
                            'balance': balance
                        })

                        # Уменьшаем позицию
                        position['volume'] = vol_in_pos - partial_volume
                        # Пропорционально уменьшаем комиссию на вход (уже учли её часть)
                        position['commission_buy'] -= position['commission_buy'] * (partial_volume / vol_in_pos)

    return balance, trade_logs

def run_experiment(future_window, rolling_window):
    df_all = make_df_all()
    df_all = df_all.drop(['futures', 'stocks'], axis=1)

    df_train, df_test = prepare_dataset_split(
        df_all,
        future_window,
        rolling_window
    )

    df_test, model_buy, model_sell = train_model_split(
        df_train,
        df_test,
        model=LinearRegression,
        need_training=True
    )

    plot_bond_prices(df_test, plot_dir=plot_dir, num_points=500)
    final_balance, trades = simulate_trading(
        df_test,
        future_window=future_window,
        threshold=0.01,
        commission_rate=0.23 / 2,
        initial_balance=100000
    )
    print("Итоговый баланс:", final_balance)
    print("Количество совершённых сделок:", len(trades))
    return df_test, None



run_experiment(future_window=150, rolling_window=20)
