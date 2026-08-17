import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handle data loading and preprocessing for time series."""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """Load and parse the dataset."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records")
        
        # Parse date column
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return self.df
    
    def inspect_data(self):
        """Perform data inspection."""
        print("\n" + "=" * 60)
        print("DATA INSPECTION")
        print("=" * 60)
        
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nData types:")
        print(self.df.dtypes)
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nStatistical summary:")
        print(self.df.describe())
        
        return self.df
    
    def handle_missing_values(self, method='ffill'):
        """Handle missing values in time series."""
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)
        
        if method == 'ffill':
            self.df = self.df.ffill()
            print("✓ Used forward fill for missing values")
        elif method == 'interpolate':
            self.df = self.df.interpolate(method='linear')
            print("✓ Used linear interpolation for missing values")
        
        print(f"Missing values after handling: {self.df.isnull().sum().sum()}")
        return self.df
    
    def detect_outliers(self, column='meantemp', method='iqr'):
        """Detect outliers in the data."""
        print("\n" + "=" * 60)
        print("OUTLIER DETECTION")
        print("=" * 60)
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | 
                              (self.df[column] > upper_bound)]
            
            print(f"Found {len(outliers)} outliers in {column}")
            print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
        return outliers
    
    def chronological_sort(self):
        """Sort data chronologically."""
        print("\n" + "=" * 60)
        print("CHRONOLOGICAL SORTING")
        print("=" * 60)
        
        self.df = self.df.sort_values('date').reset_index(drop=True)
        print("✓ Data sorted chronologically")
        return self.df