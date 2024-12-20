# Imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_forecast(
    historical_df: pd.DataFrame,
    ensemble_forecast: pd.DataFrame,
    individual_forecasts: pd.DataFrame,
    title: str = "SPY Price Forecast",
    plot_start_date: pd.Timestamp = None,
):
    """
    Create interactive visualization of the ensemble forecast
    with min and max of the forecast distribution plotted 
    as a secondary band on the primary plot.
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(
            f"{title} Forecast",
        ),
    )

    # Filter data for plotting
    if plot_start_date:
        historical_df = historical_df[historical_df['ds'] >= plot_start_date].copy()
        ensemble_forecast = ensemble_forecast[ensemble_forecast['ds'] >= plot_start_date].copy()
        individual_forecasts = individual_forecasts[individual_forecasts['ds'] >= plot_start_date].copy()

    # Set index for alignment
    ensemble_forecast.set_index('ds', inplace=True)

    # Ensemble forecast
    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat'],
            name=' Forecast',
            line=dict(color='blue', width=1)
        ),
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,255,0.5)',
            name='Upper Bound'
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=ensemble_forecast.index,
            y=ensemble_forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,255,0.5)',
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
            line=dict(color='rgba(255,165,0,1)', dash='dot'),
            name='Min Forecast'
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_max.index,
            y=forecast_max.values,
            mode='lines',
            line=dict(color='rgba(255,165,0,1)', dash='dot'),
            name='Max Forecast'
        ),
    )

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df['ds'],
            y=historical_df['y'],
            name='Historical',
            line=dict(color='black', width=0.5)
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
