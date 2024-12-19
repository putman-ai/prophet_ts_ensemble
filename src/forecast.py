# Imports
import pandas as pd
from prophet import Prophet
from typing import List, Tuple

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
