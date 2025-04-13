import joblib

from src.prints import print_metrics, print_feature_importance
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
def generate_features(df, window, diff):
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

def generate_target(df, future_window):
    df['target_buy'] = df['BondBUY'].shift(-future_window)
    df['target_sell'] = df['BondSELL'].shift(-future_window)

    return df.dropna()

def prepare_dataset_split(df_all, future_window, rolling_window):
    df_train = df_all[df_all['week'].isin([0, 1, 2, 3])].copy()
    df_test = df_all[df_all['week'].isin([4, 5, 6, 7])].copy()

    for name, df in [('train', df_train), ('test', df_test)]:
        df = calculate_bond_prices(df)
        df = generate_features(df, window=rolling_window, diff=5)
        df = generate_target(df, future_window=future_window)

        plot_bond_prices_over_time(df, './plots')
        plot_deviations_from_trend(df, './plots')

        if name == 'train':
            df_train = df
        else:
            df_test = df
    return df_train, df_test

def train_model_split(df_train, df_test, model, need_training:bool):
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
        model_sell = model()

        model_sell.fit(df_train[make_features('SELL')], df_train['target_sell'])

        df_test['PredSELL'] = model_sell.predict(df_test[make_features('SELL')])

        joblib.dump(model_sell, './models/model_sell.pkl')

    else:
        model_sell = joblib.load('./models/model_sell.pkl')

        df_test['PredSELL'] = model_sell.predict(df_test[make_features('SELL')])

    print_metrics(df_test['target_sell'], df_test['PredSELL'], "bond_sell")
    importance = print_feature_importance(model_sell, make_features('SELL'))

    return df_test, model_sell, importance