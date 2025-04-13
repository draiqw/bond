def simulate_trading(df, threshold, initial_balance=100000.0, commission_rate=0.23 / 2, future_window=50):
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
                        trade_logs.append({
                            'action': 'sell_partial',
                            'tick': tick,
                            'price_perc': bond_sell,
                            'price_rub': exit_price_rub,
                            'volume': partial_volume,
                            'commission': commission_sell,
                            'balance': balance
                        })
                        position['volume'] = vol_in_pos - partial_volume
                        position['commission_buy'] -= position['commission_buy'] * (partial_volume / vol_in_pos)

    return balance, trade_logs