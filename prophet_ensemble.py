import pandas as pd
import numpy as np
from prophet import Prophet
from typing import List, Dict, Tuple
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class ProphetEnsemble:
    def __init__(self, lookback_periods: List[int], changepoint_priors: List[float], 
                 seasonality_modes: List[str] = ['additive', 'multiplicative']):
        """
        Initialize Prophet Ensemble with different parameter combinations
        
        Args:
            lookback_periods: List of different historical periods to consider
            changepoint_priors: List of changepoint prior scale values
            seasonality_modes: List of seasonality modes to try
        """
        self.lookback_periods = lookback_periods
        self.changepoint_priors = changepoint_priors
        self.seasonality_modes = seasonality_modes
        self.models = []
        self.forecasts = []
        
    def create_models(self) -> None:
        """Create Prophet models with different parameter combinations"""
        for lookback in self.lookback_periods:
            for prior in self.changepoint_priors:
                for mode in self.seasonality_modes:
                    model = Prophet(
                        changepoint_prior_scale=prior,
                        seasonality_mode=mode,
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    self.models.append({
                        'model': model,
                        'lookback': lookback,
                        'prior': prior,
                        'mode': mode
                    })
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit all models in the ensemble
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        self.create_models()
        for model_dict in self.models:
            # Get lookback period data
            lookback_df = df.tail(model_dict['lookback'])
            # Fit model
            model_dict['model'].fit(lookback_df)
            
    def predict(self, periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts from all models and create ensemble prediction.
        Separately averages yhat, yhat_upper, and yhat_lower across all models.
        
        Args:
            periods: Number of future periods to forecast
            
        Returns:
            Tuple of (ensemble forecast, individual forecasts)
        """
        individual_forecasts = []
        
        # Generate forecasts from each model
        for model_dict in self.models:
            future = model_dict['model'].make_future_dataframe(periods=periods)
            forecast = model_dict['model'].predict(future)
            forecast['model_params'] = f"lookback:{model_dict['lookback']}_prior:{model_dict['prior']}_mode:{model_dict['mode']}"
            individual_forecasts.append(forecast)
        
        # Combine all forecasts
        all_forecasts = pd.concat(individual_forecasts, axis=0)
        
        # Create ensemble forecast by separately averaging yhat, yhat_upper, and yhat_lower
        ensemble_forecast = all_forecasts.groupby('ds').agg({
            'yhat': 'mean',
            'yhat_upper': 'mean',
            'yhat_lower': 'mean',
            'trend': 'mean'
        }).reset_index()
        
        return ensemble_forecast, all_forecasts


def plot_ensemble_forecast_with_distribution(
    historical_df: pd.DataFrame,
    ensemble_forecast: pd.DataFrame,
    individual_forecasts: pd.DataFrame,
    title: str = "SPY Price Forecast",
    plot_start_date: pd.Timestamp = None,
):
    """
    Create interactive visualization of the ensemble forecast with min and max of the forecast distribution
    plotted as a secondary band on the primary plot.
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(
            f"{title} - Ensemble Forecast",
        ),
    )

    # Filter data for plotting
    if plot_start_date:
        historical_df = historical_df[historical_df['ds'] >= plot_start_date].copy()
        ensemble_forecast = ensemble_forecast[ensemble_forecast['ds'] >= plot_start_date].copy()
        individual_forecasts = individual_forecasts[individual_forecasts['ds'] >= plot_start_date].copy()

    # Set index for alignment
    ensemble_forecast.set_index('ds', inplace=True)

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df['ds'],
            y=historical_df['y'],
            name='Historical',
            line=dict(color='black')
        ),
    )

    # Ensemble forecast
    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat'],
            name='Ensemble Forecast',
            line=dict(color='blue')
        ),
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            name='Upper Bound'
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,255,0.2)',
            name='Lower Bound'
        ),
    )

    # Min and Max of the individual forecast distribution
    forecast_distribution = individual_forecasts.pivot(
        index='ds', columns='model_params', values='yhat'
    )
    forecast_min = forecast_distribution.min(axis=1)
    forecast_max = forecast_distribution.max(axis=1)

    # Align indices for consistency
    forecast_min = forecast_min.reindex(ensemble_forecast.index)
    forecast_max = forecast_max.reindex(ensemble_forecast.index)

    # Consistency checks
    assert (ensemble_forecast['yhat'] <= forecast_max).all(), "Ensemble forecast exceeds max individual forecast!"
    assert (ensemble_forecast['yhat'] >= forecast_min).all(), "Ensemble forecast is below min individual forecast!"

    fig.add_trace(
        go.Scatter(
            x=forecast_min.index,
            y=forecast_min.values,
            mode='lines',
            line=dict(color='rgba(255,165,0,0.5)', dash='dot'),
            name='Min Forecast'
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_max.index,
            y=forecast_max.values,
            mode='lines',
            line=dict(color='rgba(255,165,0,0.5)', dash='dot'),
            name='Max Forecast'
        ),
    )

    fig.update_layout(
        height=600,
        title_x=0.5,
        showlegend=True,
        xaxis=dict(title="Date"),
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price")

    return fig

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
