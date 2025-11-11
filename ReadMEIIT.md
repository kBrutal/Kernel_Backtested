#requirements

#Files Provided
mainIIT.py
configIIT.json
dist/alpha_research-0.1.0-<tag>.whl

#Install Following 
pip install alpha_research-0.1.0-<tag>.whl

#Reinstalling alpha_research
pip install --force-reinstall alpha_research-0.1.0-cp39-cp39-linux_x86_64.whl

#Run Your Strat
python mainIIT.py configIIT.json


#configIIT Details
{
    "data_path": "/home/prashant",  (Path to data)
    "start_date": 0, 
    "end_date": 100,
    "timer": 600, (on timer callback every x seconds , it will ticker wise positon and pnl at this interval)
    "broadcast": [   
        "EBY",
        "EBX"
    ] (Will run for both EBY and EBX)
}

#Python version required
Python 3.9.23


After completion You will get all days cummulative report which includes:
Final_Equity
PnL_Rs
PnL_%
Sharpe
Sortino
Calmar
Max_Drawdown_%
Positive_Trades
Negative_Trades
Total_Trades
WinRate_%


And you will get pandas dataframe which include day wise stats for each ticker which includes :``
day
pnl
pnl_with_tc
pnl_pct"
pnl_cumsum"
max_drawdown_pct
pos_trades
neg_trades
total_trades
hit_rate_%

(ALL PNL include Penalty if there was)
If User dont square off position, it will be squared of automatically. But it will add penalty.

Placing order : only market order allowed

trade_buy = backtest.place_order(
        ticker=buy_ticker,
        qty=1,
        side=Side.BUY
    )
    # if trade_buy:
    #     print(f"Placed BUY trade: {trade_buy}")

    # place SELL
trade_sell = backtest.place_order(
    ticker=sell_ticker,
    qty=1,
    side=Side.SELL
)

trade_buy and trade_sell includes :

trade = Trade(ticker=ticker,side=side,price=exec_price,quantity=qty,timestamp=self.timestamp)

//main file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss

DON't CHANGE this MAIN function

def my_broadcast_callback(state, ts):
Here you will get all the data, state is dict, key as ticker, value as data which is again a dict. Includes Time, Price and other fields 

def on_timer(ts):
    print("On timer callback") // at 