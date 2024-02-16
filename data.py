# Import necessary libraries
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

# Define the Streamlit app title
st.set_page_config(
    page_title="Stock Portfolio Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Define the Stock class
class StockPortfolio:
    historical_prices = {}

    def __init__(self, symbol):
        """
        Initialize StockPortfolio object.

        Parameters:
            symbol (str): Stock symbol.
        """
        self.symbol = symbol
        if symbol not in StockPortfolio.historical_prices:
            data = yf.download(symbol, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
            StockPortfolio.historical_prices[symbol] = data

    def get_current_price(self, date):
        """
        Get the current price of the stock.

        Parameters:
            date (str): Date for which the price is required.

        Returns:
            float: Current price of the stock.
        """
        price = StockPortfolio.historical_prices[self.symbol].loc[date]['Adj Close'] if date in StockPortfolio.historical_prices[self.symbol].index else st.write("Data not available for the given date. Change the date to get data")
        return price

    def calculate_monthly_returns(self, date):
        """
        Calculate monthly returns of the stock.

        Parameters:
            date (str): End date of the month.

        Returns:
            float: Monthly return percentage.
        """
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=30)
        prices = StockPortfolio.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        returns = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100 if len(prices) > 0 else st.write("Data not available for the given date. Change the date to get data")
        return returns

    def calculate_daily_returns(self, date):
        """
        Calculate daily returns of the stock.

        Parameters:
            date (str): End date of the day.

        Returns:
            float: Daily return percentage.
        """
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=1)
        prices = StockPortfolio.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        returns = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100 if len(prices) > 0 else st.write("Data not available for the given date. Change the date to get data")
        return returns

    def get_last_30_days_prices(self, date):
        """
        Get prices for the last 30 days.

        Parameters:
            date (str): End date.

        Returns:
            numpy.ndarray: Array of prices for the last 30 days.
        """
        end_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=30)
        prices = StockPortfolio.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        return prices.to_numpy()

    def get_daily_returns_for_every_day(self, start_date, end_date):
        """
        Get daily returns for every day in the given period.

        Parameters:
            start_date (str): Start date of the period.
            end_date (str): End date of the period.

        Returns:
            list: List of daily returns.
        """
        prices = StockPortfolio.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        returns = [(prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1] for i in range(1, len(prices))] if len(prices) > 1 else []
        return returns

    def calculate_final_daily_return(self, daily_returns, initial_value):
        """
        Calculate the final daily return.

        Parameters:
            daily_returns (list): List of daily returns.
            initial_value (float): Initial investment value.

        Returns:
            float: Final value after applying daily returns.
        """
        final_value = initial_value
        for daily_return in daily_returns:
            final_value *= (1 + daily_return)
        return final_value   

# Function to calculate performance metrics
def calculate_performance_metrics(returns, initial_investment, start_date, end_date):
    """
    Calculate performance metrics such as CAGR, volatility, and Sharpe ratio.

    Parameters:
        returns (list): List of returns.
        initial_investment (float): Initial investment amount.
        start_date (str): Start date of the investment period.
        end_date (str): End date of the investment period.

    Returns:
        float: Compound Annual Growth Rate (CAGR).
        float: Volatility.
        float: Sharpe Ratio.
    """
    if initial_investment == 0:
        print("Error: Initial investment is zero.")
        st.error("Error: Initial investment is zero.")
        return None, None, None

    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    t = (end_datetime - start_datetime).days / 365 # Number of years

    final_value = initial_investment
    for daily_return in returns:
        final_value *= (1 + daily_return)

    CAGR = (((final_value / initial_investment) ** (1 / t)) - 1) * 100

    volatility = (np.std(returns) * np.sqrt(252)) * 100
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    return CAGR, volatility, sharpe_ratio

# Function to select top stocks based on monthly returns
def select_top_stocks(stocks, start_date, end_date):
    """
    Select top performing stocks based on monthly returns.

    Parameters:
        stocks (list): List of stock symbols.
        start_date (str): Start date of the investment period.
        end_date (str): End date of the investment period.

    Returns:
        list: List of selected top performing stock symbols.
    """
    selected_stocks = [symbol for symbol in stocks if StockPortfolio(symbol).calculate_monthly_returns(start_date) > 0]
    return selected_stocks

# Function to calculate performance metrics for selected stocks compared to benchmark
def calculate_selected_stock_performance_metrics(returns, initial_value, start_date, end_date, portfolio_dailyreturns):
    """
    Calculate performance metrics for selected stocks compared to benchmark.

    Parameters:
        returns (list): List of returns.
        initial_value (float): Initial investment amount.
        start_date (str): Start date of the investment period.
        end_date (str): End date of the investment period.
        portfolio_dailyreturns (list): List of daily returns for the portfolio.

    Returns:
        float: Compound Annual Growth Rate (CAGR).
        float: Volatility.
        float: Sharpe Ratio.
    """
    if initial_value == 0:
        print("Error: Initial value is zero.")
        st.error("Error: Initial value is zero.")
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    t = (end_datetime - start_datetime).days / 365 # Number of years
    final_value = sum(returns)   
    portfolio_dailyreturns = [(sum(sublist) / len(portfolio_dailyreturns)) for sublist in zip(*portfolio_dailyreturns)]

    CAGR = (((final_value / initial_value) ** (1 / t)) - 1) * 100

    volatility = (np.std(portfolio_dailyreturns) * np.sqrt(252)) * 100
    sharpe_ratio = (np.mean(portfolio_dailyreturns) / np.std(portfolio_dailyreturns)) * np.sqrt(252)
    return CAGR, volatility, sharpe_ratio

# Main Streamlit code
portfolio = StockPortfolio('^NSEI')

# Add instructions and input fields to the sidebar
st.sidebar.header('Portfolio Performance Analysis')
st.sidebar.subheader('Instructions')
st.sidebar.write('Enter the initial investment amount and select the start and end dates for the analysis.')

# Input fields for user in the sidebar
initial_investment = st.sidebar.number_input("Initial Investment")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Button to trigger calculation in the sidebar
if st.sidebar.button("Calculate"):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    # Define selected stocks
    stock_symbols = ['COALINDIA.NS', 'UPL.NS', 'ICICIBANK.NS', 'NTPC.NS', 'HEROMOTOCO.NS', 'AXISBANK.NS', 'HDFCLIFE.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'APOLLOHOSP.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'SBILIFE.NS', 'BRITANNIA.NS', 'SUNPHARMA.NS', 'BAJAJ-AUTO.NS', 'MARUTI.NS', 'LT.NS', 'RELIANCE.NS', 'BPCL.NS', 'TATACONSUM.NS', 'CIPLA.NS', 'M&M.NS', 'BAJFINANCE.NS', 'ITC.NS', 'ADANIPORTS.NS', 'NESTLEIND.NS', 'HDFCBANK.NS', 'DIVISLAB.NS', 'TATAMOTORS.NS', 'INDUSINDBK.NS', 'EICHERMOT.NS', 'HINDUNILVR.NS', 'ASIANPAINT.NS', 'DRREDDY.NS', 'ULTRACEMCO.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'TCS.NS', 'LTIM.NS', 'TATASTEEL.NS', 'POWERGRID.NS', 'HCLTECH.NS', 'TECHM.NS', 'INFY.NS', 'WIPRO.NS', 'JSWSTEEL.NS', 'ADANIENT.NS', 'GRASIM.NS', 'HINDALCO.NS']

    # Select top stocks
    selected_stocks = select_top_stocks(stock_symbols, start_date, end_date)

    # Calculate performance metrics for Nifty index
    CAGR, volatility, sharpe_ratio = calculate_performance_metrics(portfolio.get_daily_returns_for_every_day(start_date, end_date), initial_investment, start_date, end_date)
    nifty_metrics = {'CAGR (%)': [round(CAGR,2)], 'Volatility (%)': [round(volatility,2)], 'Sharpe Ratio': [round(sharpe_ratio,2)], "Start Date" : start_date, "End Date" : end_date}
    df_nifty_metrics = pd.DataFrame(nifty_metrics)
    st.subheader("Performance Metrics for Nifty Index")
    st.write(df_nifty_metrics)

    # Display selected stocks
    st.subheader("Selected Stocks")
    for i, symbol in enumerate(selected_stocks, start=1):
        st.write(f"{i}. {symbol}")

    # Calculate initial investment per stock
    initial_investment_per_stock = initial_investment / len(selected_stocks)

    # Calculate performance metrics for selected stocks
    portfolio_returns = []
    portfolio_dailyreturns = []
    for symbol in selected_stocks:
        stock = StockPortfolio(symbol)
        daily_returns = stock.get_daily_returns_for_every_day(start_date, end_date)
        final_return = stock.calculate_final_daily_return(daily_returns, initial_investment_per_stock)
        portfolio_returns.append(final_return)
        portfolio_dailyreturns.append(daily_returns)

    CAGR1, volatility1, sharpe_ratio1 = calculate_selected_stock_performance_metrics(portfolio_returns, initial_investment, start_date, end_date, portfolio_dailyreturns)
    selected_stocks_metrics = {'CAGR (%)': [round(CAGR1, 2)], 'Volatility (%)': [round(volatility1,2)], 'Sharpe Ratio': [round(sharpe_ratio1,2)], "Start Date" : start_date, "End Date" : end_date}
    df_selected_stocks_metrics = pd.DataFrame.from_records([selected_stocks_metrics, nifty_metrics], index=["Strategy","Benchmark"])
    st.subheader("Performance Metrics for Selected Stocks")
    st.write(df_selected_stocks_metrics)