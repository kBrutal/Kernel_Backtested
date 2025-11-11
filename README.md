# 📘 IIT Backtesting Framework

## 📂 Files Provided
- `mainIIT.py`  
- `configIIT.json`  
- `dist/alpha_research-0.1.0-<tag>.whl`  

---

## ⚙️ Installation

### 1. Install the wheel package
```bash
pip install alpha_research-0.1.0-<tag>.whl
```

### 2. (Optional) Reinstall the package
If you want to reinstall or update the same version:
```bash
pip install --force-reinstall alpha_research-0.1.0-cp39-cp39-linux_x86_64.whl
```

---

## 🚀 Running the Strategy

Execute the main script using:
```bash
python mainIIT.py configIIT.json
```

---

## 🧩 Configuration Details (`configIIT.json`)
Example:
```json
{
    "data_path": "/home/prashant",
    "start_date": 0,
    "end_date": 100,
    "timer": 600,
    "broadcast": [
        "EBY",
        "EBX"
    ]
}
```

### Explanation
| Key | Description |
|------|-------------|
| `data_path` | Absolute path to the input data directory |
| `start_date` | Starting day index for backtest |
| `end_date` | Ending day index for backtest |
| `timer` | Interval (in seconds) for periodic callback that logs positions and PnL |
| `broadcast` | List of tickers to run the backtest for (e.g., `EBY`, `EBX`) |

---

## 🐍 Python Environment
- **Python Version:** `3.9.23`

---

## 📊 Output Summary

After the backtest completes, you’ll receive:

### 1. **Cumulative Performance Report**
Includes the following metrics:
| Metric | Description |
|---------|-------------|
| `Final_Equity` | Final portfolio value |
| `PnL_Rs` | Profit or Loss in Rupees |
| `PnL_%` | Profit or Loss as a percentage |
| `Sharpe` | Sharpe ratio |
| `Sortino` | Sortino ratio |
| `Calmar` | Calmar ratio |
| `Max_Drawdown_%` | Maximum drawdown percentage |
| `Positive_Trades` | Number of profitable trades |
| `Negative_Trades` | Number of losing trades |
| `Total_Trades` | Total number of trades executed |
| `WinRate_%` | Win rate (percentage of positive trades) |

### 2. **Day-wise Ticker Statistics (`pandas.DataFrame`)**
| Column | Description |
|---------|-------------|
| `day` | Day number |
| `pnl` | Daily profit/loss |
| `pnl_with_tc` | PnL including transaction costs |
| `pnl_pct` | Daily return percentage |
| `pnl_cumsum` | Cumulative PnL |
| `max_drawdown_pct` | Max drawdown for the day |
| `pos_trades` | Number of positive trades |
| `neg_trades` | Number of negative trades |
| `total_trades` | Total trades executed |
| `hit_rate_%` | Percentage of winning trades |

> 💡 **Note:**  
> All PnL values include **penalty adjustments** (for unsquared positions, automatically squared off at the end of the day).

---

## 💼 Order Placement

Only **market orders** are supported.

### Example – Placing Trades
```python
# Place BUY
trade_buy = backtest.place_order(
    ticker=buy_ticker,
    qty=1,
    side=Side.BUY
)

# Place SELL
trade_sell = backtest.place_order(
    ticker=sell_ticker,
    qty=1,
    side=Side.SELL
)
```

Each trade object includes:
```python
Trade(
    ticker=ticker,
    side=side,
    price=exec_price,
    quantity=qty,
    timestamp=self.timestamp
)
```

---

## 🧠 Main File Structure (`mainIIT.py`)

**Do not modify the main function.**

```python
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    print(datetime.now().strftime("%H:%M:%S"))
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))
```

---

## 🔁 Callback Details

### Broadcast Callback
```python
def my_broadcast_callback(state, ts):
    # state: dict -> { ticker: { Time, Price, ... } }
    # ts: current timestamp
    ...
```
You will receive a state dictionary for each ticker containing:
- Time  
- Price  
- Other relevant fields  

### Timer Callback
```python
def on_timer(ts):
    print("On timer callback")
```
Triggered every `timer` seconds as defined in the config.

---

## ✅ Summary

**Workflow:**
1. Install dependencies  
2. Configure `configIIT.json`  
3. Run the script  
4. Review final cumulative and daily reports  
