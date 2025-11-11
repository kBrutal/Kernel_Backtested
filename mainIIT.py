import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from alpha_research import BacktesterIIT, Side
import matplotlib.pyplot as plt
from helper import KernelReturnFilter

tick_size = 15 #15 second resampling
window = 600 #window for z-score
#take profit and stoploss thresholds
tp_thresh = 0.5 # define as positive
sl_thresh = 0.3 # define as positive

#for keeping track of market position
curr_pos = 0

# trade entry variables (fed to surface on exit)
entry_price = 0.0
entry_row = None
entry_zscore = 0.0

day = 0
saved = False
day_pnl = 0.0 
last_pointer = -1

backtest_pnl = 0.0

trade_sheet = pd.DataFrame(columns=["Time", "signal", "Price"])
feature_columns = None
feature_buffer = []  
adf = pd.DataFrame() 
backtest_pnl = 0.0
last_traded_ptr = -2

# memory for a day, reset on day start
prices = []
z_scores = []

start_date = 0

#default thresholds
z_thresh_l = -1.75
z_thresh_u = 1.75
conf_thresh = 0.4

global_tc = 0.0002

PnL_of_the_day = 0.0
last_trade_PnL = 0.0

last_order_type = 0  # "BUY" or "SELL"
last_traded_price = 0.0 #for profit calculation

take_profit = 0.5
stop_loss = -0.15

min_cool_down  = 15*60 # seconds for cool down between trades

#for return surfaces
PROB_THRESH_LONG = 0.55
PROB_THRESH_SHORT = 0.55

started = False


training_day = -1
TRAIN_DAYS = 50  # Number of days to train the model
skip = False
updated = False
corrupted_days = [i for i in range(60,80)]

def safe_float(val):
    """Convert safely to float, return 0.0 on bad or NaN values."""
    try:
        f = float(val)
        if np.isnan(f):
            return 0.0
        return f
    except Exception:
        return 0.0

def discover_feature_columns_from_row(row_dict):
    """Get all T8 columns in sorted order — fixed forever once discovered."""
    return sorted([c for c in row_dict.keys() if str(c).endswith("T8")])

def make_features(df_window: pd.DataFrame) -> np.ndarray:
    """
    EXACTLY matches your flatten_data() during training:
      stats = window_data.agg(['mean','std','min','max']).values.flatten()
    """
    df_window = df_window.fillna(0.0)
    stats = df_window.agg(["mean", "std", "min", "max"]).to_numpy().flatten(order="C")
    return stats.reshape(1, -1)

def save_trade_plot(day, trade_sheet):
    os.makedirs("plots", exist_ok=True)
    data_path = f"EBX/day{day}.csv"
    if not os.path.exists(data_path):
        print(f"[Warn] {data_path} not found, skipping plot.")
        return

    df = pd.read_csv(data_path)
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("time") or cols.get("timestamp") or cols.get("t")
    price_col = cols.get("price") or cols.get("close") or cols.get("last_price")

    if time_col is None or price_col is None:
        print(f"[Warn] Missing Time/Price columns in {data_path}")
        return

    df = df[[time_col, price_col]].rename(columns={time_col: "Time", price_col: "Price"})
    try:
        df["t_sec"] = pd.to_timedelta(df["Time"]).dt.total_seconds()
    except Exception:
        df["t_sec"] = pd.to_numeric(df["Time"], errors="coerce").fillna(method="ffill")

    plt.figure(figsize=(10, 5))
    plt.plot(df["t_sec"], df["Price"], color="lightblue", linewidth=0.8, label="Price")

    if not trade_sheet.empty:
        if "Time" in trade_sheet.columns:
            trade_sheet["t_sec"] = pd.to_timedelta(trade_sheet["Time"]).dt.total_seconds()
        else:
            trade_sheet["t_sec"] = trade_sheet.index.map(lambda i: df["t_sec"].iloc[min(i, len(df)-1)])

        buy = trade_sheet.query("signal == 1")
        sell = trade_sheet.query("signal == -1")

        if not buy.empty:
            plt.scatter(buy["t_sec"], buy["Price"], color="green", marker="^", s=30, label="BUY")
        if not sell.empty:
            plt.scatter(sell["t_sec"], sell["Price"], color="red", marker="v", s=30, label="SELL")

    plt.title(f"Trades — Day {day}")
    plt.xlabel("Time (s)")
    plt.ylabel("Price")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/trade_signals_day{day}.png", dpi=120)
    plt.close()
    print(f"[Info] Saved plot for Day {day}")


FR = KernelReturnFilter(feature_dim=692) # number of rows minus one, (time and price are removed and zscore is added)

opposite_trade_active = False #for keeping track if opposite trade is active
 
resample_pointer = -1 #pointer for resampled data

def my_broadcast_callback(state, ts):
    global feature_columns, feature_buffer, adf
    global curr_pos, entry_price, saved, tick_size, last_trade_PnL, PnL_of_the_day, last_traded_ptr, prices, z_scores, trade_sheet, day, day_pnl, last_pointer, backtest_pnl
    global PROB_THRESH_LONG, PROB_THRESH_SHORT, FR
    global opposite_trade_active
    global resample_pointer
    global z_thresh_l, z_thresh_u, conf_thresh
    global last_traded_price, last_order_type
    global global_tc
    global TRAIN_DAYS
    global entry_zscore, entry_row
    global started, take_profit, stop_loss, corrupted_days, skip, updated
    global training_day, backtest_pnl

    try:
        h, m, s = map(int, ts.split(":"))
        pointer = h * 3600 + m * 60 + s
    except Exception:
        pointer = len(feature_buffer)

    tickers = [t for t, d in state.items() if d['Price'] != 0]


    # if not started:
    #     if pointer == 0:
    #         started = True
    #         updated = False  
    #     else:
    #         if not updated:
    #             print(f"[Skipping]Skipping day {day+1} due to no 00:00:00 timestamp")
    #             day += 1         
    #             updated = True
    #     return

    # else:
    #     if pointer == 0:
    #         print(f"[Skipping]Skipping day {day+1} due to no 00:00:00 timestamp")
    #         started = False       
    #         updated = False       


    if day in corrupted_days:
        if pointer == 0:
            print(f"[Skipping]Skipping corrupted day {day+1}")
            day = day + 1
        return 
    

    # initializing for each day
    if pointer == 0:
        resample_pointer = -1
        day = day+1
        training_day +=1
        prices = []
        z_scores = []
        curr_pos = 0
        tick_size = 15 # 15 seconds ticks
        last_traded_price = 0
        last_trade_time = None #unused
        last_order_type = 0  # "BUY" or "SELL"
        global_tc = 0.0002
        PnL_of_the_day = 0.0
        last_trade_PnL = 0.0
        opposite_trade_active = False
        trade_sheet = pd.DataFrame(columns=["Time", "signal", "Price"])
        last_traded_ptr = -2
        saved = False
        conf_thresh = 0.4
        z_thresh_u = 1.75
        z_thresh_l = -1.75
        entry_zscore = 0.0
        if training_day < TRAIN_DAYS:
            print(f"Day {day}: TRAINING DAY")
        else:
            print(f"Day {day}: KERNEL FILTERING DAY")
    
    if pointer%600 == 0:
        print(f"Backtest_PnL {backtest_pnl:.5f}")
        #pnl of the day
        print(f"PnL of the day {day}: {PnL_of_the_day:.5f}")

    buy_ticker = tickers[0]
    sell_ticker = tickers[0]
    price_now = 0
    signal = 0
    for key, value in state.items():
        price_now = value['Price']
        if pointer%tick_size == 0: # update prices only at tick_size intervals
            resample_pointer += 1
            if value['Price'] != 0:
                prices.append(float(value['Price'])) # using 1 second lagged prices
            else:
                prices.append(prices[-1])
            if resample_pointer<window:
                z = 0.0
                z_scores.append(z)
            else:
                past_prices = prices[resample_pointer-window:resample_pointer]
                mean_p = np.mean(past_prices)
                std_p = max(np.std(past_prices), 0.08)
                z = (price_now - mean_p) / std_p
                z_scores.append(z)
            if len(z_scores)>600:
                z_window = z_scores[-600:] 
                z_thresh_u = np.mean(z_window) + 2 * np.std(z_window)
                z_thresh_l = np.mean(z_window) - 2 * np.std(z_window)
            else:
                z_thresh_u = 1.75
                z_thresh_l = -1.75
            if z<0:
                signal_conf = abs(z + z_thresh_u)/(abs(z - z_thresh_u)+1)
            else:
                signal_conf = abs(z + z_thresh_l)/(abs(z-z_thresh_l)+1)
            signal = 0
            if (z<z_thresh_l) and curr_pos <= 0 and signal_conf > conf_thresh:
                signal = 1
                print(z, z_thresh_u, z_thresh_l, signal_conf)
                print(f"LONG signal at time {ts} with z={z:.2f} and conf={signal_conf:.2f}")

            elif (z>z_thresh_u) and curr_pos >= 0 and signal_conf > conf_thresh:
                signal = -1
                print(z, z_thresh_u, z_thresh_l, signal_conf)
                print(f"SHORT signal at time {ts} with z={z:.2f} and conf={signal_conf:.2f}")
    if pointer - last_traded_ptr == 1:
        trade_sheet.loc[len(trade_sheet)] = [ts, last_order_type, prices[-1]]

    if pointer > 6.12*3600 and curr_pos == 0: 
        if not saved:
            os.makedirs("tradesheet", exist_ok=True)
            trade_path = f"tradesheet/tradesheet_day{day}.csv"
            trade_sheet.to_csv(trade_path, index=False)
            print(f"Trade sheet saved to {trade_path}")
            save_trade_plot(day,trade_sheet)
            saved = True
        return

    row = {}
    for _, v in state.items():
        for feat, val in v.items():
            row[str(feat)] = safe_float(val)

    if not trade_sheet.empty:
        last_price = float(trade_sheet['Price'].iloc[-1])
        unrealized_PnL = (float(price_now) - last_price) * curr_pos \
                        - global_tc * abs(curr_pos) * (float(price_now) + last_price)
    else:
        unrealized_PnL = 0.0


    if (pointer - last_traded_ptr) < min_cool_down:
        return
    
    if curr_pos == 0 and PnL_of_the_day < -0.6:
        return

    PnL_of_the_day = PnL_of_the_day + unrealized_PnL

    if training_day<TRAIN_DAYS:
        if signal == 0:
            if unrealized_PnL < -sl_thresh:
                if curr_pos > 0:
                    print(f"Closing LONG position at time {ts} on Day {day} due to stop-loss")
                    returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
                    feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0
                    trade_sell = backtest.place_order(
                        ticker=sell_ticker,
                        qty=1,
                        side=Side.SELL
                    )
                    curr_pos = curr_pos - 1
                    last_traded_price = row['Price']
                    last_order_type = -1
                    last_traded_ptr = pointer
                elif curr_pos < 0:
                    print(f"Closing SHORT position at time {ts} on Day {day} due to stop-loss")
                    returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
                    feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0
                    trade_buy = backtest.place_order(
                        ticker=buy_ticker,
                        qty=1,
                        side=Side.BUY
                    )
                    curr_pos = curr_pos + 1
                    last_traded_price = row['Price']
                    last_order_type = 1
                    last_traded_ptr = pointer
        elif signal == 1:
            if curr_pos <= 0:
                trade_buy = backtest.place_order(
                    ticker=buy_ticker,
                    qty=1,
                    side=Side.BUY
                )
            if curr_pos == 0:
                print(f"Entering LONG position at time {ts} on Day {day}")
                entry_price = row['Price']
                entry_row = row
                entry_zscore = z_scores[-1]
            if curr_pos < 0:
                print(f"Closing SHORT position at time {ts} on Day {day}")
                returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
                backtest_pnl += returns
                PnL_of_the_day += returns
                entry_price = -1
                entry_row = None
                entry_zscore = 0.0
            curr_pos = curr_pos + 1
            last_traded_price = row['Price']
            last_order_type = 1
            last_traded_ptr = pointer
            
        elif signal == -1:
            if curr_pos >= 0:
                trade_sell = backtest.place_order(
                    ticker=sell_ticker,
                    qty=1,
                    side=Side.SELL
                )
            if curr_pos == 0:
                print(f"Entering SHORT position at time {ts} on Day {day}")
                entry_price = row['Price']
                entry_row = row
                entry_zscore = z_scores[-1]
            if curr_pos > 0:
                print(f"Closing LONG position at time {ts} on Day {day}")
                returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
                backtest_pnl += returns
                PnL_of_the_day += returns
                entry_price = -1
                entry_row = None
                entry_zscore = 0.0
            curr_pos = curr_pos - 1
            last_traded_price = row['Price']
            last_order_type = -1
            last_traded_ptr = pointer
        
    else:
        ABS_PROFIT = 0.6
        ABS_LOSS = 0.2
        if signal == 0:
            if curr_pos == 0:
                pass
            elif not opposite_trade_active:
                unrealized_PnL = (row['Price'] - entry_price) * curr_pos - global_tc * abs(curr_pos) * (row['Price'] + entry_price)
                if curr_pos>0 and (unrealized_PnL >= take_profit or unrealized_PnL <= -stop_loss):
                    print(f"Closing LONG position due to PnL threshold at time {ts} on Day {day}")
                    feat_vals = [z_scores[-1]] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
                    entry_price = -1
                    entry_row = None
                    trade_sell = backtest.place_order(
                        ticker=sell_ticker,
                        qty=1,
                        side=Side.SELL
                    )
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    curr_pos = curr_pos - 1
                    last_traded_price = row['Price']
                    last_order_type = -1
                    last_traded_ptr = pointer
                elif curr_pos<0 and (unrealized_PnL >= take_profit or unrealized_PnL <= -stop_loss):
                    print(f"Closing SHORT position due to PnL threshold at time {ts} on Day {day}")
                    feat_vals = [z_scores[-1]] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
                    entry_price = -1
                    entry_row = None
                    trade_buy = backtest.place_order(
                        ticker=buy_ticker,
                        qty=1,
                        side=Side.BUY
                    )
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    curr_pos = curr_pos + 1
                    last_traded_price = row['Price']
                    last_order_type = 1
                    last_traded_ptr = pointer
        elif signal == 1:
            if curr_pos == 0:
                entry_price = row['Price']
                entry_row = row
                entry_zscore = z_scores[-1]
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                prob_long = FR.predict_prob(np.array(feat_vals, dtype=np.float32), 'long')
                prob_short = FR.predict_prob(np.array(feat_vals, dtype=np.float32), 'short')
                if prob_long > PROB_THRESH_LONG:
                    print(f"LONG signal accepted with prob {prob_long:.4f} at time {ts} on Day {day}")
                    trade_buy = backtest.place_order(
                        ticker=buy_ticker,
                        qty=1,
                        side=Side.BUY
                    )
                    curr_pos = curr_pos + 1
                    last_traded_price = row['Price']
                    last_order_type = 1
                    last_traded_ptr = pointer
                elif prob_short > 1000:
                    print(f"LONG signal REJECTED with prob {prob_long:.4f} at time {ts} on Day {day}, going SHORT instead with prob {prob_short:.4f}")
                    entry_price = row['Price']
                    entry_row = row
                    entry_zscore = z_scores[-1]
                    opposite_trade_active = True
                    trade_sell = backtest.place_order(
                        ticker=sell_ticker,
                        qty=1,
                        side=Side.SELL
                    )
                    curr_pos = curr_pos - 1
                    last_traded_price = row['Price']
                    last_order_type = -1
                    last_traded_ptr = pointer
                else:
                    print(f"LONG signal REJECTED with prob {prob_long:.4f} at time {ts} on Day {day}, and no opposite trade taken")
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0

            elif curr_pos < 0 and not opposite_trade_active:
                print(f"Closing accepted SHORT position at time {ts} on Day {day}")
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
                FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
                entry_price = -1
                entry_row = None
                entry_zscore = 0.0
                trade_buy = backtest.place_order(
                    ticker=buy_ticker,
                    qty=1,
                    side=Side.BUY
                )
                backtest_pnl += returns
                PnL_of_the_day += returns
                curr_pos = curr_pos + 1
                last_traded_price = row['Price']
                last_order_type = +1
                last_traded_ptr = pointer
            
        elif signal == -1:
            if curr_pos == 0:
                entry_price = row['Price']
                entry_row = row
                entry_zscore = z_scores[-1]
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                prob_short = FR.predict_prob(np.array(feat_vals, dtype=np.float32), 'short')
                prob_long = FR.predict_prob(np.array(feat_vals, dtype=np.float32), 'long')
                if prob_short > PROB_THRESH_SHORT:
                    print(f"SHORT signal accepted with prob {prob_short:.4f} at time {ts} on Day {day}")
                    trade_sell = backtest.place_order(
                        ticker=sell_ticker,
                        qty=1,
                        side=Side.SELL
                    )
                    curr_pos = curr_pos - 1
                    last_traded_price = row['Price']
                    last_order_type = -1
                    last_traded_ptr = pointer
                elif prob_long > 1000:
                    print(f"SHORT signal REJECTED with prob {prob_short:.4f} at time {ts} on Day {day}, going LONG instead with prob {prob_long:.4f}")
                    opposite_trade_active = True
                    trade_buy = backtest.place_order(
                        ticker=buy_ticker,
                        qty=1,
                        side=Side.BUY
                    )
                    curr_pos = curr_pos + 1
                    last_traded_price = row['Price']
                    last_order_type = 1
                    last_traded_ptr = pointer
                else:
                    print(f"SHORT signal REJECTED with prob {prob_short:.4f} at time {ts} on Day {day}, and no opposite trade taken")
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0
            elif curr_pos > 0 and not opposite_trade_active:
                print(f"Closing accepted LONG position at time {ts} on Day {day}")
                feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
                FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
                entry_price = -1
                entry_row = None
                entry_zscore = 0.0
                trade_sell = backtest.place_order(
                    ticker=sell_ticker,
                    qty=1,
                    side=Side.SELL
                )
                backtest_pnl += returns
                PnL_of_the_day += returns
                curr_pos = curr_pos - 1
                last_traded_price = row['Price']
                last_order_type = -1
                last_traded_ptr = pointer
        
        if opposite_trade_active:
            unrealized_PnL = (float(price_now) - entry_price) * curr_pos - global_tc * abs(curr_pos) * (float(price_now) + entry_price)
            if unrealized_PnL >= ABS_PROFIT or unrealized_PnL <= -ABS_LOSS:
                print(f"Day {day}: Closing rejected position due to PnL threshold")
                if curr_pos > 0:
                    trade_sell = backtest.place_order(
                        ticker=sell_ticker,
                        qty=1,
                        side=Side.SELL
                    )
                    feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0
                    curr_pos = curr_pos - 1
                    last_order_type = -1
                    last_traded_ptr = pointer
                elif curr_pos < 0:
                    trade_buy = backtest.place_order(
                        ticker=buy_ticker,
                        qty=1,
                        side=Side.BUY
                    )
                    feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
                    returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
                    FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
                    backtest_pnl += returns
                    PnL_of_the_day += returns
                    entry_price = -1
                    entry_row = None
                    entry_zscore = 0.0
                    curr_pos = curr_pos + 1
                    last_order_type = 1
                    last_traded_ptr = pointer
                opposite_trade_active = False
        
        

    if pointer >= 23324:  # Forced Square-OFF at 6:28:44 AM
        if curr_pos > 0:
            trade_sell = backtest.place_order(
                ticker=sell_ticker,
                qty=curr_pos,
                side=Side.SELL
            )
            returns = (row['Price'] - entry_price) - global_tc * (entry_price + row['Price'])
            feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
            FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'long')
            backtest_pnl += returns
            PnL_of_the_day += returns
            entry_price = -1
            entry_row = None
            entry_zscore = 0.0
            curr_pos = 0
            last_order_type = -1
            last_traded_ptr = pointer
            print(f"Forced SQUARE-OFF SELL at {ts} for Day {day}")
        elif curr_pos < 0:
            trade_buy = backtest.place_order(
                ticker=buy_ticker,
                qty=abs(curr_pos),
                side=Side.BUY
            )
            returns = (entry_price - row['Price']) - global_tc * (entry_price + row['Price'])
            feat_vals = [entry_zscore] + [safe_float(x) for x in list(entry_row.values())[2:]]
            FR.update(np.array(feat_vals, dtype=np.float32), float(returns), 'short')
            backtest_pnl += returns
            PnL_of_the_day += returns
            entry_price = -1
            entry_row = None
            entry_zscore = 0.0
            curr_pos = 0
            last_order_type = 1
            last_traded_ptr = pointer
            print(f"Forced SQUARE-OFF BUY at {ts} for Day {day}")

    if pointer > 23324 and curr_pos == 0 and not saved:
        os.makedirs("tradesheet", exist_ok=True)
        os.makedirs("feature_logs", exist_ok=True)
        trade_path = f"tradesheet/tradesheet_day{day}.csv"
        # adf_path = f"feature_logs/features_day{day}.csv"
        trade_sheet.to_csv(trade_path, index=False)
        # adf.to_csv(adf_path, index=False)
        save_trade_plot(day, trade_sheet)
        saved = True
        print(f"Saved tradesheet: {trade_path}")
        print(f"backtest PnL: {backtest_pnl:.5f}")
        print(f"PnL of the day {day}: {PnL_of_the_day:.5f}")
        # print(f"Saved features:   {adf_path}")


def on_timer(ts):
    pass
