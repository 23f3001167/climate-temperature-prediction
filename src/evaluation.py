import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

class ModelEvaluator:
    """Evaluate and compare models."""
    
    def __init__(self):
        self.comparison_df = None
    
    def compare_models(self, rf_metrics, lstm_metrics):
        """Compare Random Forest and LSTM models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        self.comparison_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²', 'MAPE'],
            'Random Forest': [
                rf_metrics['mae'],
                rf_metrics['rmse'],
                rf_metrics['r2'],
                rf_metrics['mape']
            ],
            'LSTM': [
                lstm_metrics['mae'],
                lstm_metrics['rmse'],
                lstm_metrics['r2'],
                lstm_metrics['mape']
            ]
        })
        
        # Calculate improvement
        self.comparison_df['Improvement'] = (
            (self.comparison_df['LSTM'] - self.comparison_df['Random Forest']) / 
            self.comparison_df['Random Forest'] * 100
        ).round(2)
        
        print(self.comparison_df.to_string(index=False))
        return self.comparison_df
    
    def save_comparison(self, path):
        """Save comparison to file."""
        self.comparison_df.to_csv(path, index=False)
        print(f"✓ Comparison saved to: {path}")
    
    def plot_results(self, rf_model, lstm_model, X_test, y_test, plot_dir):
        """Generate visualization plots."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Make predictions
        rf_pred = rf_model.predict(X_test)
        
        # Get LSTM predictions (need to handle differently)
        # Assuming lstm_model has evaluate method
        lstm_eval = lstm_model.evaluate()
        lstm_pred = lstm_eval['y_pred'].flatten()
        
        # 1. Actual vs Predicted
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest
        axes[0].scatter(y_test, rf_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Temperature')
        axes[0].set_ylabel('Predicted Temperature')
        axes[0].set_title('Random Forest: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # LSTM
        axes[1].scatter(y_test.values[-len(lstm_pred):], lstm_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Temperature')
        axes[1].set_ylabel('Predicted Temperature')
        axes[1].set_title('LSTM: Actual vs Predicted')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created: actual_vs_predicted.png")
        
        # 2. Time series prediction
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get test indices
        test_indices = y_test.index
        
        # Random Forest
        axes[0].plot(test_indices, y_test, label='Actual', alpha=0.7)
        axes[0].plot(test_indices, rf_pred, label='Predicted', alpha=0.7)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Temperature')
        axes[0].set_title('Random Forest: Time Series Prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # LSTM
        lstm_test_indices = test_indices[-len(lstm_pred):]
        axes[1].plot(lstm_test_indices, y_test.values[-len(lstm_pred):], label='Actual', alpha=0.7)
        axes[1].plot(lstm_test_indices, lstm_pred, label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Temperature')
        axes[1].set_title('LSTM: Time Series Prediction')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'time_series_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created: time_series_prediction.png")
        
        # 3. Residual plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest residuals
        rf_residuals = y_test - rf_pred
        axes[0].scatter(rf_pred, rf_residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Temperature')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Random Forest: Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # LSTM residuals
        lstm_residuals = y_test.values[-len(lstm_pred):] - lstm_pred
        axes[1].scatter(lstm_pred, lstm_residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Temperature')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('LSTM: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'residual_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created: residual_plots.png")
        
        # 4. Feature importance (Random Forest)
        if hasattr(rf_model, 'feature_importance') and rf_model.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = rf_model.feature_importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.title('Random Forest: Top 15 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plot_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Created: feature_importance.png")
    
    def generate_forecasts(self, rf_model, lstm_model, X, y, forecasts_dir):
        """Generate future forecasts."""
        print("\n" + "=" * 60)
        print("GENERATING FORECASTS")
        print("=" * 60)
        
        # Get last sequence for LSTM
        last_sequence = X.values[-30:]
        
        # Forecast next 30 days
        rf_future = self._forecast_rf(rf_model, X, days=30)
        lstm_future = lstm_model.predict_future(last_sequence, days=30)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'day': range(1, 31),
            'random_forest_forecast': rf_future,
            'lstm_forecast': lstm_future
        })
        
        # Save forecast
        forecast_path = forecasts_dir / 'future_forecast.csv'
        forecast_df.to_csv(forecast_path, index=False)
        print(f"✓ Forecast saved to: {forecast_path}")
        print(forecast_df.head(10))
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 31), rf_future, label='Random Forest', marker='o')
        plt.plot(range(1, 31), lstm_future, label='LSTM', marker='s')
        plt.xlabel('Day')
        plt.ylabel('Temperature (°C)')
        plt.title('30-Day Temperature Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(forecasts_dir / 'future_forecast_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created: future_forecast_plot.png")
        
        return forecast_df
    
    def _forecast_rf(self, rf_model, X, days=30):
        """Generate forecast using Random Forest."""
        # Simple forecast: use last available features
        last_features = X.iloc[-1:].copy()
        forecasts = []
        
        for i in range(days):
            pred = rf_model.predict(last_features)[0]
            forecasts.append(pred)
            
            # Update features for next iteration (simplified)
            # This is a simplified approach - in practice, you'd need to update all features
            last_features = last_features.copy()
            # Shift values or use predictions as new inputs
        
        return np.array(forecasts)