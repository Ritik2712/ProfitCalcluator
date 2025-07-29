import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def profitCalculator(avg1, avg2, df, price_column, name):
    fast, slow = (avg2, avg1) if avg1 > avg2 else (avg1, avg2)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='ignore')
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ['Close', 'Volume', 'Dividends', 'Stock Splits', 'High', 'Low']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Calculate SMAs
    df['SMA Fast'] = df[price_column].rolling(window=fast, min_periods=1).mean()
    df['SMA Slow'] = df[price_column].rolling(window=slow, min_periods=1).mean()

    # Generate signals
    df['Signal'] = np.where(df['SMA Fast'] > df['SMA Slow'], 'Buy', 'Sell')

    # Initialize tracking variables
    number_of_stocks = 0
    purchased_Amount = 0
    profit_loss = 0
    sell_markers = []
    buy_markers = []

    for index, row in df.iterrows():
        # Buy condition: SMA fast > SMA slow and currently no stock
        if row['Signal'] == "Buy" and number_of_stocks == 0:
            number_of_stocks = 1
            purchased_Amount = row[price_column]
            buy_markers.append((row['Date'], row['SMA Fast']))

        # Sell condition: SMA fast <= SMA slow and currently holding stock
        elif row['Signal'] == "Sell" and number_of_stocks > 0:
            curr_profit = (row[price_column] * number_of_stocks) - purchased_Amount
            profit_loss += curr_profit
            sell_markers.append((row['Date'], row['SMA Fast']))

            # Reset position after selling
            number_of_stocks = 0
            purchased_Amount = 0

        # Track cumulative profit
        df.loc[index, 'Profit/Loss'] = profit_loss

    # If stock is still held at the end, sell it at the last price
    if number_of_stocks > 0:
        profit_loss += (df.loc[len(df)-1, price_column] * number_of_stocks) - purchased_Amount

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['SMA Fast'], label=f'SMA {fast}', color='blue')
    ax.plot(df['Date'], df['SMA Slow'], label=f'SMA {slow}', color='orange')

    # Add markers
    if buy_markers:
        buy_dates, buy_prices = zip(*buy_markers)
        ax.scatter(buy_dates, buy_prices, color='green', label='Buy', marker='^', s=80)
    if sell_markers:
        sell_dates, sell_prices = zip(*sell_markers)
        ax.scatter(sell_dates, sell_prices, color='red', label='Sell', marker='v', s=80)

    ax.set_title(f'SMA {fast} vs SMA {slow} with Buy/Sell Points for {name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    return df, profit_loss, fig


# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.title("SMA Profit Calculator")

st.markdown("""
Analyze how a **Simple Moving Average (SMA) crossover strategy** 
would have performed on a stock in the past.

**Strategy:**  
- Buy once when short-term SMA crosses above long-term SMA (and you have no stock).  
- Sell when short-term SMA crosses below long-term SMA (if you have stock).
""")

stock_symbol = st.text_input(
    "Stock Symbol",
    placeholder="e.g., AAPL (Apple) or INFY.NS (Infosys)",
    help="This is the short code for a stock. Example: AAPL for Apple, INFY.NS for Infosys on NSE."
)

sma1 = st.number_input(
    "Short-term Moving Average (SMA 1)",
    min_value=1,
    value=10,
    help="Number of days for the short-term moving average (tracks quick price changes)."
)

sma2 = st.number_input(
    "Long-term Moving Average (SMA 2)",
    min_value=1,
    value=70,
    help="Number of days for the long-term moving average (tracks long-term trends)."
)

if st.button("Calculate Profit"):
    if stock_symbol.strip() == "":
        st.warning("Please enter a stock symbol before running the calculation.")
    else:
        ticker = yf.Ticker(stock_symbol)
        df = ticker.history(period="2y")
        df.reset_index(inplace=True)
        result_df, profit, fig = profitCalculator(sma1, sma2, df, 'Open', stock_symbol)

        st.success(f"Total Profit/Loss: ₹ {round(profit, 2)}")
        st.pyplot(fig)

        st.subheader("Detailed Results")
        st.dataframe(result_df)
