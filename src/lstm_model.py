import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    """LSTM model for time series forecasting."""
    
    def __init__(self, X, y, sequence_length=30):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # LSTM parameters
        self.params = {
            'lstm_units': [64, 32],
            'dropout': 0.2,
            'dense_units': 16,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.1
        }
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, test_size=0.2):
        """Split data with time series ordering (NO RANDOM SHUFFLE)."""
        print("\n" + "=" * 60)
        print("LSTM TIME SERIES SPLIT")
        print("=" * 60)
        
        # Scale the data
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_X.fit_transform(self.X)
        y_scaled = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1)).flatten()
        
        # Split without shuffling
        split_index = int(len(X_scaled) * (1 - test_size))
        
        X_train_scaled = X_scaled[:split_index]
        y_train_scaled = y_scaled[:split_index]
        X_test_scaled = X_scaled[split_index:]
        y_test_scaled = y_scaled[split_index:]
        
        # Create sequences
        self.X_train, self.y_train = self.create_sequences(X_train_scaled, y_train_scaled)
        self.X_test, self.y_test = self.create_sequences(X_test_scaled, y_test_scaled)
        
        print(f"✓ Training sequences: {len(self.X_train)}")
        print(f"✓ Test sequences: {len(self.X_test)}")
        print(f"✓ Sequence length: {self.sequence_length}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, input_shape):
        """Build the LSTM model."""
        print("\n" + "=" * 60)
        print("LSTM MODEL ARCHITECTURE")
        print("=" * 60)
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.params['lstm_units'][0], 
            return_sequences=True, 
            input_shape=input_shape
        ))
        model.add(Dropout(self.params['dropout']))
        
        # Second LSTM layer
        if len(self.params['lstm_units']) > 1:
            model.add(LSTM(self.params['lstm_units'][1], return_sequences=False))
            model.add(Dropout(self.params['dropout']))
        
        # Dense layers
        model.add(Dense(self.params['dense_units'], activation='relu'))
        model.add(Dropout(self.params['dropout']))
        model.add(Dense(1))
        
        # Compile
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(model.summary())
        
        self.model = model
        return model
    
    def train(self):
        """Train the LSTM model."""
        print("\n" + "=" * 60)
        print("LSTM TRAINING")
        print("=" * 60)
        
        # Build model
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        self.build_model(input_shape)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=self.params['validation_split'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✓ LSTM model trained successfully")
        print(f"✓ Best validation loss: {min(history.history['val_loss']):.4f}")
        
        return history
    
    def evaluate(self):
        """Evaluate the LSTM model."""
        print("\n" + "=" * 60)
        print("LSTM EVALUATION")
        print("=" * 60)
        
        # Predict
        y_pred_scaled = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_actual = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'y_pred': y_pred,
            'y_actual': y_actual
        }
    
    def predict_future(self, last_sequence, days=30):
        """Predict future days."""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next day
            pred_scaled = self.model.predict(current_sequence.reshape(1, self.sequence_length, -1), verbose=0)
            predictions.append(pred_scaled[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_scaled[0]
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
    
    def save_model(self, path):
        """Save the model and scalers."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
            'params': self.params
        }, path.with_suffix('.pkl'))
        
        print(f"✓ LSTM model saved to: {path}")
    
    def load_model(self, path):
        """Load the model and scalers."""
        from tensorflow.keras.models import load_model
        
        self.model = load_model(path)
        data = joblib.load(path.with_suffix('.pkl'))
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.sequence_length = data['sequence_length']
        self.params = data['params']
        
        print(f"✓ LSTM model loaded from: {path}")