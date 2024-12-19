# Prophet Ensemble Forecasting

A sophisticated time series forecasting implementation that addresses common challenges with Facebook Prophet through ensemble methods. This project combines multiple Prophet models with varying parameters to create more robust and reliable forecasts, particularly focusing on capturing tail events and reducing sensitivity to historical lookback periods.

## Overview

The Prophet Ensemble Forecaster improves upon standard Prophet implementations by:
1. Reducing parameter sensitivity through ensemble methods
2. Better capturing of tail events through varied changepoint priors
3. Adapting to different seasonality patterns using multiple seasonality modes
4. Creating more robust predictions through varied historical lookback periods

## Implementation Details

### Core Components

- **ProphetEnsemble Class**: Manages multiple Prophet models with different configurations
- **Parameter Combinations**:
  - Lookback periods: Multiple training windows (5-10 years)
  - Changepoint priors: Range from 0.001 to 0.5
  - Seasonality modes: Both additive and multiplicative
- **Ensemble Method**: Averages predictions across all models for final forecast
- **Visualization**: Interactive Plotly charts showing ensemble prediction with confidence intervals

### Key Features

- Automated model creation with different parameter combinations
- Separate averaging of prediction intervals (yhat, yhat_upper, yhat_lower)
- Distribution visualization showing min/max ranges across all models
- Configurable prediction periods and lookback windows
- Built-in visualization tools with historical context

## Technical Details

```python
class ProphetEnsemble:
    def __init__(self, lookback_periods, changepoint_priors, seasonality_modes):
        # Initialize ensemble with parameter combinations
        
    def create_models(self):
        # Create Prophet models with different parameters
        
    def fit(self, df):
        # Fit all models in ensemble
        
    def predict(self, periods):
        # Generate and combine forecasts
```

### Dependencies

- pandas
- numpy
- prophet
- yfinance (for data fetching)
- plotly (for visualization)

## Suggested Improvements

### Model Enhancements

1. **Extended Ensemble Methods**
   - Integration with ARIMA models for short-term accuracy
   - LSTM integration for complex pattern recognition
   - XGBoost for capturing non-linear relationships
   - Weighted ensemble based on model performance

2. **Validation and Testing**
   - Walk-forward optimization
   - Rolling window backtesting
   - Out-of-sample validation
   - Cross-validation for parameter tuning
   - Model performance metrics (MAPE, RMSE, MAE)

3. **Advanced Features**
   - Dynamic weight adjustment based on model performance
   - Automatic parameter optimization
   - Anomaly detection in historical data
   - Confidence interval calibration
   - External regressor support

4. **Technical Improvements**
   - Parallel processing for model fitting
   - Memory optimization for large datasets
   - Caching mechanism for intermediate results
   - API wrapper for easy integration

### Performance Measurement

1. **Backtesting Framework**
   - Multiple evaluation windows
   - Performance comparison with benchmark models
   - Statistical significance testing
   - Market regime analysis

2. **Metrics Suite**
   - Traditional metrics (MAPE, RMSE, MAE)
   - Financial metrics (Sharpe ratio, maximum drawdown)
   - Directional accuracy
   - Tail event capture rate

3. **Validation Methods**
   - K-fold time series cross-validation
   - Out-of-time validation
   - Sensitivity analysis
   - Monte Carlo simulations

## Usage

```python
# Initialize ensemble
ensemble = ProphetEnsemble(
    lookback_periods=[1260, 1512, 1764, 2016, 2268, 2520],  # 5-10 years
    changepoint_priors=[0.001, 0.01, 0.1, 0.5],
    seasonality_modes=['additive', 'multiplicative']
)

# Fit models and generate predictions
ensemble.fit(df)
forecast, individual_forecasts = ensemble.predict(periods=30)

# Visualize results
fig = plot_ensemble_forecast_with_distribution(
    historical_df=df,
    ensemble_forecast=forecast,
    individual_forecasts=individual_forecasts
)
```

## Contributing

Contributions are welcome! Areas of particular interest:
- Implementation of suggested improvements
- Performance optimization
- Additional visualization options
- Extended validation methods
- Documentation improvements

## License

MIT License

## Citation

If you use this code in your research or project, please cite:

```
@misc{prophet-ensemble,
  author = {John Putman II},
  title = {Prophet Ensemble Forecasting},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/putman-ai/prophet_ts_ensemble}
}
```

## Acknowledgments

This project builds upon Facebook's Prophet forecasting tool and incorporates ensemble methods inspired by various forecasting literature and best practices in time series analysis.

