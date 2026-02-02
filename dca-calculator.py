import yfinance as yf
import pandas as pd
from datetime import datetime

def calculate_dca(ticker, amount, frequency, start_date):
    """
    Calculates the outcome of a Dollar Cost Averaging strategy.

    Parameters:
    - ticker (str): Stock symbol (e.g., 'AAPL', 'VOO', 'BTC-USD').
    - amount (float): Amount of money invested per interval.
    - frequency (str): 'daily', 'weekly', 'bi-weekly', 'monthly', 'quarterly', 'yearly'.
    - start_date (str): Format 'YYYY-MM-DD'.

    Returns:
    - dict: A summary of results (Total Invested, Portfolio Value, ROI, etc.).
    - pd.DataFrame: A detailed log of every buy transaction.
    """
    
    # 1. Map frequency strings to Pandas offset aliases
    freq_map = {
        'daily': 'B',        # Business day
        'weekly': 'W',       # Weekly
        'bi-weekly': '2W',   # Every 2 weeks
        'monthly': 'MS',     # Month Start
        'quarterly': 'QS',   # Quarter Start
        'yearly': 'YS'       # Year Start
    }
    
    if frequency not in freq_map:
        raise ValueError(f"Invalid frequency. Choose from: {list(freq_map.keys())}")

    # 2. Fetch Historical Data
    # We fetch a bit extra data to ensure we have the latest close price
    print(f"Fetching data for {ticker}...")
    try:
        # auto_adjust=True ensures we get the Split/Dividend adjusted price 
        # which is usually preferred for long-term backtesting.
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    except Exception as e:
        return {"Error": f"Failed to download data: {e}"}, None

    if data.empty:
        return {"Error": "No data found. Check ticker or start date."}, None

    # Handle MultiIndex columns if yfinance returns them (common in newer versions)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, level=1, axis=1)

    # 3. Generate Ideal Buy Dates
    # Create a range of dates from start to yesterday (to avoid future buys if running early in the day)
    end_date = datetime.now()
    ideal_dates = pd.date_range(start=start_date, end=end_date, freq=freq_map[frequency])

    # 4. Align Buy Dates with Actual Trading Days
    # We reindex the stock data to our ideal dates. 
    # method='bfill' (backfill) means: If buy date is Saturday, use Monday's price.
    # We assume we buy at the 'Close' price of that day.
    buy_log = data['Close'].reindex(ideal_dates, method='bfill')
    
    # Remove any NaNs (dates where no future data exists yet or data is missing)
    buy_log = buy_log.dropna()
    
    # Convert Series to DataFrame for calculation
    df_dca = pd.DataFrame(buy_log)
    df_dca.columns = ['Price']
    
    # 5. Perform Calculations
    df_dca['Invested'] = amount
    df_dca['Shares_Purchased'] = df_dca['Invested'] / df_dca['Price']
    
    # Cumulative stats
    df_dca['Total_Shares'] = df_dca['Shares_Purchased'].cumsum()
    df_dca['Total_Invested'] = df_dca['Invested'].cumsum()
    
    # Final Metrics
    total_invested = df_dca['Invested'].sum()
    total_shares = df_dca['Shares_Purchased'].sum()
    
    # Get the absolute latest price available (Current Market Price)
    # We re-fetch the last 5 days history specifically to get the most recent print
    ticker_obj = yf.Ticker(ticker)
    todays_data = ticker_obj.history(period='5d')
    if not todays_data.empty:
        current_price = todays_data['Close'].iloc[-1]
    else:
        # Fallback to last row of downloaded history
        current_price = data['Close'].iloc[-1]

    portfolio_value = total_shares * current_price
    profit_loss = portfolio_value - total_invested
    roi = (profit_loss / total_invested) * 100 if total_invested > 0 else 0

    # 6. Construct Summary
    summary = {
        "Ticker": ticker,
        "Start Date": start_date,
        "Frequency": frequency,
        "Total Invested": round(total_invested, 2),
        "Total Shares Owned": round(total_shares, 4),
        "Average Cost Basis": round(total_invested / total_shares, 2) if total_shares > 0 else 0,
        "Current Share Price": round(current_price, 2),
        "Portfolio Value": round(portfolio_value, 2),
        "Profit/Loss": round(profit_loss, 2),
        "ROI (%)": f"{round(roi, 2)}%"
    }

    return summary, df_dca

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    ticker_symbol = "FTEC"
    invest_amount = 4000
    freq = "monthly" 
    start = "2020-01-01"

    stats, log = calculate_dca(ticker_symbol, invest_amount, freq, start)

    if "Error" in stats:
        print(stats)
    else:
        print("-" * 30)
        print("DCA CALCULATION RESULTS")
        print("-" * 30)
        for k, v in stats.items():
            print(f"{k}: {v}")
            
        print("\nFirst 5 Transactions:")
        print(log.head())