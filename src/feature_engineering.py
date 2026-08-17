import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    """Create features for time series forecasting."""
    
    def __init__(self, df, target_col='meantemp'):
        self.df = df.copy()
        self.target_col = target_col
        self.original_cols = df.columns.tolist()
    
    def create_temporal_features(self):
        """Create temporal features from date."""
        print("\n" + "=" * 60)
        print("TEMPORAL FEATURES")
        print("=" * 60)
        
        # Ensure date is datetime
        if 'date' in self.df.columns:
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day'] = self.df['date'].dt.day
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_year'] = self.df['date'].dt.dayofyear
            
            # Cyclical encoding for month (sin/cos)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            
            # Cyclical encoding for day of week
            self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
            
            print("✓ Created temporal features:")
            print(f"  - year, month, day, day_of_week, quarter, day_of_year")
            print(f"  - month_sin, month_cos (cyclical encoding)")
            print(f"  - day_of_week_sin, day_of_week_cos (cyclical encoding)")
        
        return self.df
    
    def create_lag_features(self, lags=[1, 2, 3, 7, 14, 30]):
        """Create lag features for time series."""
        print("\n" + "=" * 60)
        print("LAG FEATURES")
        print("=" * 60)
        
        for lag in lags:
            self.df[f'{self.target_col}_lag_{lag}'] = self.df[self.target_col].shift(lag)
        
        print(f"✓ Created lag features: {lags}")
        return self.df
    
    def create_rolling_features(self, windows=[7, 14, 30]):
        """Create rolling statistics features."""
        print("\n" + "=" * 60)
        print("ROLLING FEATURES")
        print("=" * 60)
        
        for window in windows:
            self.df[f'{self.target_col}_rolling_mean_{window}'] = self.df[self.target_col].rolling(window=window).mean()
            self.df[f'{self.target_col}_rolling_std_{window}'] = self.df[self.target_col].rolling(window=window).std()
            self.df[f'{self.target_col}_rolling_min_{window}'] = self.df[self.target_col].rolling(window=window).min()
            self.df[f'{self.target_col}_rolling_max_{window}'] = self.df[self.target_col].rolling(window=window).max()
        
        print(f"✓ Created rolling features for windows: {windows}")
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features between weather variables."""
        print("\n" + "=" * 60)
        print("INTERACTION FEATURES")
        print("=" * 60)
        
        if 'humidity' in self.df.columns and 'wind_speed' in self.df.columns:
            self.df['humidity_wind_interaction'] = self.df['humidity'] * self.df['wind_speed']
            print("✓ Created: humidity * wind_speed")
        
        if 'humidity' in self.df.columns and 'meanpressure' in self.df.columns:
            self.df['humidity_pressure_interaction'] = self.df['humidity'] * self.df['meanpressure']
            print("✓ Created: humidity * pressure")
        
        return self.df
    
    def prepare_features(self, feature_cols=None, target_col='meantemp'):
        """Prepare features and target for modeling."""
        print("\n" + "=" * 60)
        print("PREPARING FEATURES")
        print("=" * 60)
        
        # Drop rows with NaN (from lag and rolling features)
        self.df = self.df.dropna()
        
        # Select features
        if feature_cols is None:
            # Use all columns except date and target
            exclude_cols = ['date', target_col]
            feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Number of features: {len(X.columns)}")
        
        return X, y, feature_cols
    
    def get_feature_columns(self):
        """Return all feature columns."""
        exclude_cols = ['date', self.target_col]
        return [col for col in self.df.columns if col not in exclude_cols]