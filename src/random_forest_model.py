import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RandomForestModel:
    """Random Forest model for time series forecasting."""
    
    def __init__(self, X, y, test_size=0.2):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importance = None
        
        # Model parameters
        self.params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def split_data(self, test_size=None):
        """Split data with time series ordering (NO RANDOM SHUFFLE)."""
        print("\n" + "=" * 60)
        print("TIME SERIES SPLIT")
        print("=" * 60)
        
        if test_size is None:
            test_size = self.test_size
        
        # Get split index
        split_index = int(len(self.X) * (1 - test_size))
        
        # Split without shuffling
        self.X_train = self.X.iloc[:split_index]
        self.y_train = self.y.iloc[:split_index]
        self.X_test = self.X.iloc[split_index:]
        self.y_test = self.y.iloc[split_index:]
        
        print(f"✓ Training data: {len(self.X_train)} samples")
        print(f"✓ Test data: {len(self.X_test)} samples")
        print(f"✓ Split date: {self.X.index[split_index] if hasattr(self.X, 'index') else split_index}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """Train the Random Forest model."""
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING")
        print("=" * 60)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Train model
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train_scaled, self.y_train)
        
        print("✓ Model trained successfully")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 10 Features ---")
        print(self.feature_importance.head(10))
        
        return self.model
    
    def evaluate(self):
        """Evaluate the model."""
        print("\n" + "=" * 60)
        print("RANDOM FOREST EVALUATION")
        print("=" * 60)
        
        # Scale test data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.scaler.transform(self.X_train), 
            self.y_train, 
            cv=5, 
            scoring='r2'
        )
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path):
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'params': self.params
        }, path)
        print(f"✓ Model saved to: {path}")
    
    def load_model(self, path):
        """Load the model."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']
        self.params = data['params']
        print(f"✓ Model loaded from: {path}")