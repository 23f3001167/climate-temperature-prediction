# src/main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# Import modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from random_forest_model import RandomForestModel
from lstm_model import LSTMModel
from evaluation import ModelEvaluator

# Set paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "DailyDelhiClimateTrain.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "DailyDelhiClimateTest.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"

# Create directories
for directory in [MODELS_DIR, PLOTS_DIR, METRICS_DIR, FORECASTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def check_data_files():
    """Check if data files exist."""
    print("\n" + "=" * 60)
    print("CHECKING DATA FILES")
    print("=" * 60)
    
    if DATA_PATH.exists():
        print(f"✅ Train data found: {DATA_PATH}")
        print(f"   Size: {DATA_PATH.stat().st_size / 1024:.2f} KB")
    else:
        print(f"❌ Train data NOT found: {DATA_PATH}")
        print("Please download DailyDelhiClimateTrain.csv from Kaggle")
        return False
    
    if TEST_PATH.exists():
        print(f"✅ Test data found: {TEST_PATH}")
        print(f"   Size: {TEST_PATH.stat().st_size / 1024:.2f} KB")
    else:
        print(f"⚠️ Test data not found: {TEST_PATH} (optional)")
    
    return True

def main():
    print("=" * 60)
    print("🌡️ CLIMATE / TEMPERATURE PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Project Directory: {BASE_DIR}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Check data files
    if not check_data_files():
        print("\n❌ Cannot proceed without data files.")
        print("Please download from: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data")
        return
    
    try:
        # 1. Load and preprocess data
        print("\n" + "=" * 60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        preprocessor = DataPreprocessor(DATA_PATH)
        df = preprocessor.load_data()
        df = preprocessor.inspect_data()
        df = preprocessor.handle_missing_values(method='ffill')
        df = preprocessor.chronological_sort()
        
        print(f"\n✓ Data preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        
        # 2. Feature engineering
        print("\n" + "=" * 60)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 60)
        
        engineer = FeatureEngineer(df, target_col='meantemp')
        df = engineer.create_temporal_features()
        df = engineer.create_lag_features(lags=[1, 2, 3, 7, 14, 30])
        df = engineer.create_rolling_features(windows=[7, 14, 30])
        df = engineer.create_interaction_features()
        
        print(f"\n✓ Feature engineering complete: {len(df.columns)} total columns")
        
        # 3. Prepare features
        print("\n" + "=" * 60)
        print("STEP 3: PREPARING FEATURES")
        print("=" * 60)
        
        X, y, feature_cols = engineer.prepare_features()
        print(f"✓ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # 4. Random Forest Model
        print("\n" + "=" * 60)
        print("STEP 4: RANDOM FOREST MODEL")
        print("=" * 60)
        
        rf_model = RandomForestModel(X, y, test_size=0.2)
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf_model.split_data()
        rf_model.train()
        rf_metrics = rf_model.evaluate()
        rf_model.save_model(MODELS_DIR / "random_forest_model.pkl")
        
        print("\n✓ Random Forest model complete")
        
        # 5. LSTM Model
        print("\n" + "=" * 60)
        print("STEP 5: LSTM MODEL")
        print("=" * 60)
        print("Note: LSTM training may take 3-5 minutes...")
        
        lstm_model = LSTMModel(X, y, sequence_length=30)
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = lstm_model.split_data(test_size=0.2)
        lstm_model.train()
        lstm_metrics = lstm_model.evaluate()
        lstm_model.save_model(MODELS_DIR / "lstm_model.h5")
        
        print("\n✓ LSTM model complete")
        
        # 6. Compare models
        print("\n" + "=" * 60)
        print("STEP 6: MODEL COMPARISON")
        print("=" * 60)
        
        evaluator = ModelEvaluator()
        comparison_df = evaluator.compare_models(rf_metrics, lstm_metrics)
        evaluator.save_comparison(METRICS_DIR / "model_comparison.txt")
        
        print("\n✓ Model comparison complete")
        
        # 7. Visualize results
        print("\n" + "=" * 60)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        evaluator.plot_results(
            rf_model, lstm_model, 
            rf_model.X_test, rf_model.y_test,
            PLOTS_DIR
        )
        
        print("\n✓ Visualizations generated")
        
        # 8. Generate forecasts
        print("\n" + "=" * 60)
        print("STEP 8: GENERATING FUTURE FORECASTS")
        print("=" * 60)
        
        evaluator.generate_forecasts(
            rf_model, lstm_model,
            X, y,
            FORECASTS_DIR
        )
        
        print("\n✓ Forecasts generated")
        
        # 9. Summary
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"{'Metric':<15} {'Random Forest':<15} {'LSTM':<15} {'Improvement':<15}")
        print("-" * 65)
        for _, row in comparison_df.iterrows():
            metric = row['Metric']
            rf_val = row['Random Forest']
            lstm_val = row['LSTM']
            imp = row['Improvement']
            print(f"{metric:<15} {rf_val:<15.4f} {lstm_val:<15.4f} {imp:>+14.2f}%")
        
        print("\n📁 GENERATED OUTPUTS:")
        print("-" * 50)
        print(f"  • Random Forest Model: {MODELS_DIR / 'random_forest_model.pkl'}")
        print(f"  • LSTM Model: {MODELS_DIR / 'lstm_model.h5'}")
        print(f"  • Plots: {PLOTS_DIR}/")
        print(f"  • Metrics: {METRICS_DIR}/")
        print(f"  • Forecasts: {FORECASTS_DIR}/")
        print(f"  • Comparison: {METRICS_DIR / 'model_comparison.txt'}")
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS:")
        print("-" * 50)
        
        # Determine which model is better
        if lstm_metrics['r2'] > rf_metrics['r2']:
            print("  ✅ LSTM performs better for this time series data")
            print("  📈 Consider using LSTM for production forecasts")
        else:
            print("  ✅ Random Forest performs well for this dataset")
            print("  📈 Consider using Random Forest for production")
        
        print("\n  💡 For better results:")
        print("     - Increase LSTM epochs (100 → 200)")
        print("     - Add more lag features (60, 90 days)")
        print("     - Try XGBoost as another baseline")
        
        print("\n" + "=" * 60)
        print("🚀 PIPELINE COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that all imports are working")
        print("2. Verify data file exists and is not corrupted")
        print("3. Make sure all dependencies are installed")
        print("4. Try running: python -c 'import tensorflow; print(tensorflow.__version__)'")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()