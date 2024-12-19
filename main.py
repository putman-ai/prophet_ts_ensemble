import pandas as pd
import yfinance as yf
from src.forecast import ProphetEnsemble
from src.visualization import plot_ensemble_forecast_with_distribution

if __name__ == "__main__":
    # Fetch SPY data
    ticker = "SPY"
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=10)  # Fetch 10 years of data
    
    spy = yf.download(ticker, start=start_date, end=end_date)
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': spy.index,
        'y': spy['Adj Close']
    }).reset_index(drop=True)
    
    # Convert lookback periods from years to days
    trading_days_per_year = 252
    lookback_years = range(5, 11)  # 5 to 10 years
    lookback_periods = [years * trading_days_per_year for years in lookback_years]
    
    # Create ensemble with different parameters
    ensemble = ProphetEnsemble(
        lookback_periods=lookback_periods,
        changepoint_priors=[0.001, 0.01, 0.1, 0.5],
        seasonality_modes=['additive', 'multiplicative']
    )
    
    # Fit ensemble and generate predictions
    ensemble.fit(df)
    forecast, individual_forecasts = ensemble.predict(periods=30)
    
    # Set plot start date to 5 years ago (matching shortest lookback)
    plot_start_date = end_date - pd.DateOffset(years=5)
    
    # Create and display visualization
    fig = plot_ensemble_forecast_with_distribution(
        historical_df=df, 
        ensemble_forecast=forecast, 
        individual_forecasts=individual_forecasts, 
        title=f"{ticker} Price Forecast",
        plot_start_date=plot_start_date
    )
    fig.show()
