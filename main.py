from mainIIT import my_broadcast_callback, on_timer
from alpha_research import BacktesterIIT, Side

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 mainv4.py <config.json>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    import json

    with open(cfg_path, "r") as f:
        config = json.load(f)

    start_date = int(config.get("start_date", 408))
    end_date = int(config.get("end_date", 509))
    data_path = config.get("data_path", "./")

    # === INITIALIZE GLOBAL DAY BASED ON CONFIG ===
    day = start_date - 1
    print(f"Initialized from config → start_date={start_date}, end_date={end_date}, initial day={day}")

    # === Initialize backtester ===
    backtest = BacktesterIIT(cfg_path)
    print(datetime.now().strftime("%H:%M:%S"))
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))