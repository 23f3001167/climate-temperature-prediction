# 🌡️ Climate/Temperature Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

A comprehensive time-series forecasting system that predicts future temperatures using both classical Machine Learning (Random Forest) and Deep Learning (LSTM) approaches.

## 🎯 Key Results

| Model | R² Score | MAE | RMSE | MAPE |
|-------|----------|-----|------|------|
| **Random Forest** | **0.9527** | **0.98°C** | **1.24°C** | **3.50%** |
| LSTM | 0.8807 | 1.65°C | 2.04°C | 5.87% |

🏆 **Winner: Random Forest** - Better accuracy, lower error

## 📊 Dataset

**Source**: [Delhi Daily Climate Dataset](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

**Features**:
- `date` - Date (chronological)
- `meantemp` - Mean temperature (Target)
- `humidity` - Humidity percentage
- `wind_speed` - Wind speed
- `meanpressure` - Atmospheric pressure

## 🏗️ Feature Engineering

| Feature Type | Examples |
|--------------|----------|
| **Temporal** | year, month, day, day_of_week, quarter, day_of_year, cyclical encoding |
| **Lag** | 1, 2, 3, 7, 14, 30 days |
| **Rolling** | mean, std, min, max for 7, 14, 30 days |
| **Interaction** | humidity × wind_speed, humidity × pressure |

## 🚀 Installation & Setup

```bash
# Clone repository
git clone https://github.com/23f3001167/climate-temperature-prediction.git
cd climate-temperature-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place in data/raw/DailyDelhiClimateTrain.csv

# Run pipeline
python src/main.py