import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

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

    df.dropna(inplace=True)
    return df

def calculate_bond_prices(df):
    df['BondBUY'] = (df['BID_F_P0'] / df['OFFER_S_P0'] - 1) * 100
    df['BondSELL'] = (df['OFFER_F_P0'] / df['BID_S_P0'] - 1) * 100
    df['VolumeBUY'] = df[['BID_F_Q0', 'OFFER_S_Q0']].min(axis=1)
    df['VolumeSELL'] = df[['OFFER_F_Q0', 'BID_S_Q0']].min(axis=1)

    return df.dropna()

def generate_targets(
        df,
        horizon=1000,
        commission_threshold=0.23
):
    """
    Для каждой строки t ищем первый k (1 <= k <= horizon),
    при котором BondSELL_(t+k) <= BondBUY_t - commission_threshold.
    Если такой k найден:
       target_class = 1
       target_time = k
    Иначе:
       target_class = 0
       target_time = horizon  (или можно NaN и исключить)
    """
    # Создаём пустые массивы под таргеты
    target_class = np.zeros(len(df), dtype=int)
    target_time = np.full(len(df), horizon, dtype=int)

    bond_buy_arr = df['BondBUY'].values
    bond_sell_arr = df['BondSELL'].values
    n = len(df)

    for i in range(n):
        # Ищем впереди в пределах horizon
        current_buy_price = bond_buy_arr[i]
        # Желаемое целевое значение BondSELL
        target_sell_value = current_buy_price - commission_threshold
        end_idx = min(i + horizon, n-1)

        # Перебираем от i+1 до end_idx
        found_k = False
        for k in range(1, end_idx - i + 1):
            if bond_sell_arr[i + k] < target_sell_value:
                target_class[i] = 1
                target_time[i] = k
                found_k = True
                break

        # Если ничего не нашли, оставляем 0 и horizon
        # if not found_k:  # по умолчанию итак 0 и horizon
        #     pass

    df['target_class'] = target_class
    df['target_time'] = target_time
    return df

def train_two_models(df_train, df_test, need_fit=True):
    """
    1. Модель классификации (LogisticRegression) -> target_class
    2. Модель регрессии (LinearRegression) -> target_time
    """
    features = [
        'BondBUY', 'BondSELL',
        'TrendSELL', 'MomentumSELL', 'RollingSTD_SELL', 'DiffFromTrendSELL',
        'TrendBUY', 'MomentumBUY', 'RollingSTD_BUY', 'DiffFromTrendBUY',
        'VolumeBUY', 'VolumeSELL'
    ]

    X_train = df_train[features].values
    y_class_train = df_train['target_class'].values
    y_time_train = df_train['target_time'].values

    X_test = df_test[features].values
    y_class_test = df_test['target_class'].values
    y_time_test = df_test['target_time'].values

    if need_fit:
        model_class = LogisticRegression(class_weight='balanced', max_iter=1000)

        model_time = LinearRegression()

        model_class.fit(X_train, y_class_train)
        model_time.fit(X_train, y_time_train)

        joblib.dump(model_class, 'model_class.pkl')
        joblib.dump(model_time, 'model_time.pkl')

    else:
        model_class = joblib.load('model_class.pkl')
        model_time = joblib.load('model_time.pkl')

    y_class_pred = model_class.predict(X_test)
    y_class_prob = model_class.predict_proba(X_test)[:,1]
    y_time_pred = model_time.predict(X_test)

    acc = accuracy_score(y_class_test, y_class_pred)
    mae_time = mean_absolute_error(y_time_test, y_time_pred)

    print("===== Результаты классификации =====")
    print("Accuracy =", acc)
    print(classification_report(y_class_test, y_class_pred))
    print("===== Результаты регрессии =====")
    print("MAE по времени до события =", mae_time)

    df_test['pred_class'] = y_class_pred
    df_test['pred_class_prob'] = y_class_prob
    df_test['pred_time'] = y_time_pred

    return df_test, model_class, model_time

def prepare_dataset_split(
    df_all,
    rolling_window=20,
    horizon=1000,
    commission_threshold=0.23
):
    df_train = df_all[df_all['week'].isin([0, 1])].copy()
    df_test = df_all[df_all['week'].isin([2, 3, 4])].copy()

    # Функция обёртка
    def transform(df):
        df = calculate_bond_prices(df)
        df = generate_features(
            df,
            window=rolling_window,
            diff=5
        )
        df = generate_targets(
            df,
            horizon=horizon,
            commission_threshold=commission_threshold
        )
        return df

    df_train = transform(df_train)
    df_test = transform(df_test)

    return df_train, df_test

def simulate_trading(
    df,
    prob_threshold=0.6,
    initial_balance=100000.0,
    commission_rate=0.23 / 2,
    k = 1.5
):
    balance = initial_balance
    position = None
    trade_logs = []

    n = len(df)

    for i in range(n):
        row = df.iloc[i]

        bond_buy = row['BondBUY']
        bond_sell = row['BondSELL']
        offer_stock = row['OFFER_S_P0']  # для вычисления bond_buy_rub
        bid_stock = row['BID_S_P0']      # для вычисления bond_sell_rub
        volume_buy = row['VolumeBUY']
        volume_sell = row['VolumeSELL']

        # Предсказанные моделью значения
        p_success = row['pred_class_prob']  # вероятность успеха от логистической регрессии
        future_window = int(row['pred_time']) if row['pred_time'] > 0 else 1

        # Считаем "рублёвую" стоимость
        bond_buy_rub = bond_buy * offer_stock / 100.0
        bond_sell_rub = bond_sell * bid_stock / 100.0

        # Фильтруем некорректные значения
        if (bond_buy <= 0) or (bond_sell <= 0):
            continue

        # ================== Если нет позиции, пробуем открыть ==================
        if position is None:
            if p_success > prob_threshold:
                # Рассчитываем, сколько можем купить
                # Сколько "бумаг" можем взять исходя из рублевого баланса и rub-стоимости одной штуки
                cand_volume = int(balance // offer_stock)  # учтём и комиссию на одну шт.
                volume = min(volume_buy, cand_volume)
                if volume > 0:
                    cost = bond_buy_rub * volume
                    commission_buy = volume * commission_rate
                    total_cost = cost + commission_buy

                    if balance >= total_cost:
                        balance -= total_cost
                        position = {
                            'entry_tick': i,
                            'entry_price': bond_buy,   # В процентах, для контроля логики
                            'entry_price_rub': bond_buy_rub,
                            'volume': volume,
                            'commission_buy': commission_buy,
                            'planned_close': i + future_window
                        }
                        trade_logs.append({
                            'action': 'open',
                            'tick': i,
                            'bond_buy_perc': bond_buy,
                            'bond_buy_rub': bond_buy_rub,
                            'volume': volume,
                            'commission_buy': commission_buy,
                            'balance_after_open': balance
                        })

        # ================== Если уже есть позиция, возможно, закрываем ==================
        else:
            ticks_in_position = i - position['entry_tick']
            # Условие «цена BondSELL опустилась ниже (entry_price - комиссия)» в процентах
            # ИЛИ вышли за предел "planned_close"
            # (Напомним, шорт: выигрыш = Покупка(выше) - Продажа(ниже), минус комиссия).
            # Но тут можно варьировать условие (например, если bond_sell < entry_price - 0.23),
            # либо проверять i >= planned_close.
            if (bond_sell < k*(position['entry_price'] - 0.23)) or (i >= position['planned_close']):
                vol_in_pos = position['volume']

                # Если можем закрыть всю позицию сразу (volume_sell >= весь наш объём)
                if volume_sell >= vol_in_pos:
                    proceeds = bond_sell * vol_in_pos  # в процентах
                    # Финальные деньги = proceeds - комиссия_продажи
                    commission_sell = vol_in_pos * commission_rate
                    # Прибыль в процентах
                    profit_perc = (position['entry_price'] - bond_sell) * vol_in_pos
                    profit_perc -= (position['commission_buy'] + commission_sell)

                    # Если хотим перевести profit из процентов в рубли, можно умножать на соответствующую цену акции,
                    # но здесь упрощённо продолжаем работать "в условных единицах".
                    # Для упрощения (как вы делали в своем коде) можно взять:
                    #   proceeds_rub = bond_sell_rub * vol_in_pos
                    #   profit_rub = (position['entry_price_rub'] - bond_sell_rub)*vol_in_pos - (position['commission_buy'] + commission_sell)
                    # Но ниже оставлен вариант "в процентах".

                    balance += proceeds - commission_sell
                    trade_logs.append({
                        'action': 'close',
                        'tick': i,
                        'bond_sell_perc': bond_sell,
                        'volume': vol_in_pos,
                        'commission_sell': commission_sell,
                        'profit_perc': profit_perc,  # чистая прибыль в процентах
                        'balance_after_close': balance
                    })
                    position = None

                else:
                    # Закрываем только часть (volume_sell), если доступно меньше
                    partial_volume = volume_sell
                    if partial_volume > 0:
                        # Пропорция закрываемой части
                        portion = partial_volume / vol_in_pos

                        # Считаем пропорциональную продажу
                        proceeds_partial = bond_sell * partial_volume
                        commission_sell_partial = partial_volume * commission_rate

                        profit_perc_partial = (position['entry_price'] - bond_sell) * partial_volume
                        profit_perc_partial -= (position['commission_buy'] * portion + commission_sell_partial)

                        balance += proceeds_partial - commission_sell_partial

                        trade_logs.append({
                            'action': 'close_partial',
                            'tick': i,
                            'bond_sell_perc': bond_sell,
                            'partial_volume': partial_volume,
                            'commission_sell_partial': commission_sell_partial,
                            'profit_partial_perc': profit_perc_partial,
                            'balance_after_close': balance
                        })

                        # Обновляем остаток позиции
                        position['volume'] = vol_in_pos - partial_volume
                        # Пропорционально «считаем израсходованную комиссию на покупку»
                        position['commission_buy'] -= position['commission_buy'] * portion

                    # Если volume_sell == 0, придётся ждать следующего тика (i+1), пока не сможем закрыть сделку
    return balance, trade_logs


def run_experiment(
    horizon=1000,
    rolling_window=20,
    prob_threshold=0.6,
    need_training=True
):
    # 1. Читаем общий датасет
    df_all = make_df_all()  # Замените на реальную реализацию
    # 2. Разделяем train/test, генерируем фичи и таргеты
    df_train, df_test = prepare_dataset_split(
        df_all,
        rolling_window=rolling_window,
        horizon=horizon,
        commission_threshold=0.23  # общая комиссия на сделку (0.23)
    )
    # 3. Обучаем 2 модели
    df_test, model_class, model_time = train_two_models(df_train, df_test, need_fit=need_training)


    for threshold in [0.91, 0.92, 0.93, 0.94, 0.96, 0.98]:
        final_balance, trades = simulate_trading(
            df_test,
            prob_threshold=threshold,
            commission_rate=0.23 / 2,
            initial_balance=100000
        )
        print(
            f"threshold={threshold},"
            f"final_balance={final_balance},"
            f"trades={len(trades)}"
        )

    print(f"Итоговый баланс = {final_balance:.2f}")
    print(f"Количество сделок (открытий/закрытий) = {len(trades)}")
    if trades:
        print("Пример логов последних 5 сделок:")
        for t in trades[-5:]:
            print(t)

    return df_test, model_class, model_time, final_balance, trades



run_experiment(rolling_window=20)
