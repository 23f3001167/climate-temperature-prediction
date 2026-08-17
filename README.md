# 🏠 House Rent Prediction System

An end-to-end **Machine Learning regression project** for predicting monthly house rental prices in major Indian cities using a **Random Forest Regressor**.

The project covers the complete machine learning lifecycle: **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, cross-validation, hyperparameter tuning, evaluation, model persistence, and predictions on unseen properties**.

🔗 **Repository:** [House Rent Prediction System](https://github.com/23f3001167/house-rent-prediction)

---

## 📌 Overview

Accurately estimating house rent depends on several factors, including property size, number of bedrooms, location, furnishing status, floor information, and number of bathrooms.

This project builds a reproducible machine learning pipeline using the **India House Rent Prediction Dataset** and trains a **Random Forest Regressor** to predict monthly rental prices.

### ✨ Key Achievements

- 📊 Processed and analyzed **4,746 rental property records**
- 🏙️ Covered **6 major Indian cities**
- 🛠️ Created **8+ engineered features**
- 🌲 Built baseline and hyperparameter-tuned **Random Forest** models
- 🔍 Used **RandomizedSearchCV** for hyperparameter optimization
- 🔄 Performed **5-fold cross-validation**
- 📈 Generated **13 analytical and model-evaluation visualizations**
- 💾 Persisted trained models using **Joblib**
- 🏠 Added prediction support for unseen properties
- 🧪 Included automated tests for the ML pipeline
- 🎯 Achieved a tuned **R² score of 0.7619**
- 📉 Achieved an **MAE of ₹8,647**

---

## 🚀 Features

### Core Machine Learning Features

- Automated dataset loading and inspection
- Missing-value handling
- Categorical feature processing
- Exploratory Data Analysis
- Feature engineering
- Train/test splitting
- Random Forest regression
- Hyperparameter tuning
- 5-fold cross-validation
- Model evaluation
- Feature importance analysis
- Residual analysis
- Model persistence
- Batch prediction on unseen properties
- Prediction output export

### 🔬 Technical Features

- Reusable preprocessing pipeline
- Numerical and categorical feature handling
- Engineered floor-related attributes
- Nonlinear size transformations
- Randomized hyperparameter search
- Multiple regression evaluation metrics
- Baseline vs tuned model comparison
- Visualization generation
- Serialized `.pkl` model files
- CSV-based batch inference
- Unit testing support

---

# 📊 Dataset

## Dataset Source

The project uses the **India House Rent Prediction Dataset from Kaggle**.

| Attribute | Value |
|---|---|
| Number of Records | 4,746 |
| Number of Features | 12 |
| Target Variable | `Rent` |
| Problem Type | Regression |
| Geographic Coverage | 6 Cities |

### 🏙️ Cities Covered

- Kolkata
- Mumbai
- Bangalore
- Delhi
- Chennai
- Hyderabad

---

## 📋 Dataset Features

| Feature | Description |
|---|---|
| `BHK` | Number of bedrooms, hall, and kitchen |
| `Rent` | Monthly house rent — target variable |
| `Size` | Property size |
| `Floor` | Floor on which the property is located and total floors |
| `Area Type` | Type of property area |
| `Area Locality` | Locality of the property |
| `City` | City where the property is located |
| `Furnishing Status` | Furnished, semi-furnished, or unfurnished |
| `Tenant Preferred` | Preferred tenant category |
| `Bathroom` | Number of bathrooms |
| `Point of Contact` | Contact type for the property |
| `Posted On` | Date on which the property listing was posted |

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Testing | Pytest |
| Model | Random Forest Regressor |
| Hyperparameter Search | RandomizedSearchCV |
| Validation | 5-Fold Cross-Validation |

---

# 🧠 Feature Engineering

Raw property information is transformed into additional features designed to expose useful relationships between property characteristics and rental price.

### Engineered Features

| Feature | Purpose |
|---|---|
| `Floor_Number` | Extracts the property's current floor |
| `Total_Floors` | Extracts the total number of floors |
| `Is_Ground` | Indicates whether the property is on the ground floor |
| `Floor_Ratio` | Represents relative floor position |
| `Size_per_BHK` | Measures property area available per BHK |
| `Room_Bathroom_Ratio` | Captures the relationship between rooms and bathrooms |
| `Log_Size` | Log transformation of property size |
| `Size_Squared` | Captures nonlinear relationships involving size |

These derived attributes allow the model to capture patterns that may not be directly represented by the original dataset columns.

---

# 🏗️ Technical Architecture

## Machine Learning Pipeline

```text
                     ┌──────────────────────┐
                     │ House Rent Dataset   │
                     │       (CSV)          │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Data Loading &       │
                     │ Initial Inspection   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Data Cleaning &      │
                     │ Preprocessing        │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Exploratory Data     │
                     │ Analysis (EDA)       │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Feature Engineering  │
                     │ 8+ Derived Features  │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Feature Encoding &   │
                     │ Transformation       │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Train / Test Split   │
                     └──────────┬───────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ Random Forest Regressor    │
                  │      Baseline Model        │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ 5-Fold Cross-Validation    │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ RandomizedSearchCV         │
                  │ Hyperparameter Tuning      │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ Tuned Random Forest Model  │
                  └─────────────┬──────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
     ┌────────────────────┐           ┌────────────────────┐
     │ Model Evaluation   │           │ Model Persistence  │
     │ R² / MAE / RMSE    │           │      Joblib        │
     │ MAPE / Accuracy    │           └─────────┬──────────┘
     └──────────┬─────────┘                     │
                │                               ▼
                │                    ┌────────────────────┐
                │                    │ Unseen Property    │
                │                    │ Predictions        │
                │                    └─────────┬──────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Results, CSVs &      │
                    │ Visualizations       │
                    └──────────────────────┘
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/23f3001167/house-rent-prediction.git
cd house-rent-prediction
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Verify the Dataset

Ensure the dataset exists at:

```text
data/raw/House_Rent_Dataset.csv
```

## 5. Run the Complete Pipeline

```bash
python src/main.py
```

The pipeline performs preprocessing, feature engineering, training, evaluation, hyperparameter tuning, visualization generation, model persistence, and prediction on unseen property records.

---

# 🚀 Usage Guide

## Run the ML Pipeline

From the project root directory:

```bash
python src/main.py
```

Generated artifacts are stored primarily in:

```text
models/
outputs/
sample/
```

---

## 🔮 Make Predictions

The project includes sample unseen property records at:

```text
sample/sample_unseen_houses.csv
```

Run the main pipeline:

```bash
python src/main.py
```

Predictions generated by the tuned model are written to:

```text
sample/predictions_output_tuned.csv
```

---

# 🔮 Sample Predictions

Example predictions generated for unseen property records:

| BHK | Size | City | Predicted Rent |
|---:|---:|---|---:|
| 2 | 850 | Mumbai | ₹46,280 |
| 3 | 1500 | Bangalore | ₹58,139 |
| 1 | 550 | Pune | ₹7,456 |
| 2 | 1200 | Chennai | ₹44,413 |
| 3 | 1800 | Delhi | ₹32,711 |

> Prediction values are model-generated estimates and should not be interpreted as guaranteed market rental prices.

---

# 📈 Model Performance

The final tuned Random Forest model achieved the following results:

| Metric | Result |
|---|---:|
| **R² Score** | **0.7619** |
| **MAE** | **₹8,647** |
| **RMSE** | **₹20,214** |
| **MAPE** | **28.53%** |
| **Accuracy** | **91.55%** |
| **Cross-Validation R²** | **0.6972 ± 0.0237** |
| **Best CV R²** | **0.7902** |

---

## 📊 Metric Interpretation

### R² Score — 0.7619

The coefficient of determination measures the proportion of variance in rental prices explained by the model.

An R² of **0.7619** indicates that the model explains approximately **76.19% of the variance** in the evaluated rental-price data.

### Mean Absolute Error — ₹8,647

The model's predictions differ from the actual rental values by approximately **₹8,647 on average**, based on MAE.

### Root Mean Squared Error — ₹20,214

RMSE penalizes larger prediction errors more heavily than MAE. The achieved RMSE is **₹20,214**.

### Mean Absolute Percentage Error — 28.53%

The model achieved a **MAPE of 28.53%** on the evaluated data.

### Reported Accuracy — 91.55%

The project reports an **accuracy value of 91.55%** according to the project's evaluation calculation.

> For regression problems, metrics such as **R², MAE, RMSE, and MAPE** are generally more informative than classification-style accuracy.

### Cross-Validation

The model achieved:

```text
Cross-Validation R² = 0.6972 ± 0.0237
Best CV R²          = 0.7902
```

Five-fold cross-validation provides a broader estimate of model performance across multiple data partitions rather than relying exclusively on one train/test split.

---

# 🔧 Hyperparameter Tuning

The project uses **RandomizedSearchCV** to explore combinations of Random Forest hyperparameters efficiently.

## Optimal Hyperparameters

| Hyperparameter | Optimal Value |
|---|---:|
| `n_estimators` | `300` |
| `max_depth` | `20` |
| `min_samples_split` | `2` |
| `min_samples_leaf` | `1` |
| `max_features` | `None` |

The tuned configuration balances predictive performance with model complexity.

---

# 📊 Results & Visualizations

The project generates visualizations for both exploratory analysis and model evaluation.

## 1. Rent Distribution

**File:** `outputs/rent_distribution.png`

Shows the overall distribution and range of monthly rental prices.

```markdown
![Rent Distribution](outputs/rent_distribution.png)
```

## 2. Rent by BHK

**File:** `outputs/rent_by_bhk.png`

Compares rental-price distributions across different BHK configurations.

```markdown
![Rent by BHK](outputs/rent_by_bhk.png)
```

## 3. Rent by City

**File:** `outputs/rent_by_city.png`

Shows how rental prices vary across the cities represented in the training dataset.

```markdown
![Rent by City](outputs/rent_by_city.png)
```

## 4. Rent by Furnishing Status

**File:** `outputs/rent_by_furnishing.png`

Compares rental prices for furnished, semi-furnished, and unfurnished properties.

```markdown
![Rent by Furnishing](outputs/rent_by_furnishing.png)
```

## 5. Correlation Matrix

**File:** `outputs/correlation_matrix.png`

Displays relationships among available numerical and engineered variables.

```markdown
![Correlation Matrix](outputs/correlation_matrix.png)
```

## 6. Rent vs Property Size

**File:** `outputs/rent_vs_size.png`

Visualizes the relationship between property size and monthly rent.

```markdown
![Rent vs Size](outputs/rent_vs_size.png)
```

---

## 🌲 Baseline Model Analysis

### 7. Baseline Actual vs Predicted

**File:** `outputs/baseline_actual_vs_predicted.png`

Compares actual rental prices against predictions produced by the baseline Random Forest model.

```markdown
![Baseline Actual vs Predicted](outputs/baseline_actual_vs_predicted.png)
```

### 8. Baseline Residual Plot

**File:** `outputs/baseline_residual_plot.png`

Visualizes baseline-model prediction errors to help identify systematic error patterns.

```markdown
![Baseline Residual Plot](outputs/baseline_residual_plot.png)
```

### 9. Baseline Feature Importance

**File:** `outputs/baseline_feature_importance.png`

Displays the top 20 features used by the baseline Random Forest model.

```markdown
![Baseline Feature Importance](outputs/baseline_feature_importance.png)
```

---

## 🎯 Tuned Model Analysis

### 10. Tuned Actual vs Predicted

**File:** `outputs/tuned_actual_vs_predicted.png`

Shows actual rental prices against predictions generated by the hyperparameter-tuned model.

```markdown
![Tuned Actual vs Predicted](outputs/tuned_actual_vs_predicted.png)
```

### 11. Tuned Residual Plot

**File:** `outputs/tuned_residual_plot.png`

Provides residual analysis for the final tuned model.

```markdown
![Tuned Residual Plot](outputs/tuned_residual_plot.png)
```

### 12. Tuned Feature Importance

**File:** `outputs/tuned_feature_importance.png`

Displays the 20 most influential features identified by the tuned Random Forest.

```markdown
![Tuned Feature Importance](outputs/tuned_feature_importance.png)
```

### 13. Model Comparison

**File:** `outputs/model_comparison_plot.png`

Provides a side-by-side visual comparison of baseline and tuned model performance.

```markdown
![Model Comparison](outputs/model_comparison_plot.png)
```

---

# 🧪 Testing

Automated tests are stored in:

```text
tests/test_main.py
```

Testing helps verify preprocessing, feature engineering, prediction behavior, input handling, and other reusable components of the ML workflow.

## Run All Tests

```bash
pytest tests/ -v
```

## Run Tests with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

After execution, open the generated coverage report:

```text
htmlcov/index.html
```

This provides a detailed view of which portions of the source code are covered by automated tests.

---

# 📁 Project Structure

```text
house-rent-prediction/
│
├── data/
│   └── raw/
│       └── House_Rent_Dataset.csv
│
├── models/
│   ├── random_forest_baseline.pkl
│   └── random_forest_model_tuned.pkl
│
├── outputs/
│   ├── rent_distribution.png
│   ├── rent_by_bhk.png
│   ├── rent_by_city.png
│   ├── rent_by_furnishing.png
│   ├── correlation_matrix.png
│   ├── rent_vs_size.png
│   ├── baseline_actual_vs_predicted.png
│   ├── baseline_residual_plot.png
│   ├── baseline_feature_importance.png
│   ├── tuned_actual_vs_predicted.png
│   ├── tuned_residual_plot.png
│   ├── tuned_feature_importance.png
│   ├── model_comparison_plot.png
│   └── *.txt
│
├── sample/
│   ├── sample_unseen_houses.csv
│   └── predictions_output_tuned.csv
│
├── src/
│   └── main.py
│
├── tests/
│   └── test_main.py
│
├── requirements.txt
└── README.md
```

---

# 💡 Technical Choices

## 🌲 Why Random Forest Regressor?

Random Forest was selected because it is well suited to structured/tabular regression problems and can capture complex nonlinear relationships between property characteristics and rent.

Key reasons include:

- **Nonlinear modelling:** Rental prices do not necessarily change linearly with size, BHK, floor, or location.
- **Feature interactions:** Random Forest can learn interactions among several property characteristics.
- **Robustness:** Combining many decision trees generally provides more stable predictions than relying on a single tree.
- **Limited preprocessing requirements:** Tree-based models do not require numerical features to be scaled in the same way as many distance- or gradient-based algorithms.
- **Feature importance:** The trained model provides feature-importance values useful for model interpretation.
- **Strong tabular-data performance:** Random Forest provides a reliable baseline and final model for this type of structured regression problem.

---

## 🧩 Why Feature Engineering?

Raw data does not always expose the relationships most useful to a machine learning algorithm.

For example, the original `Floor` field contains information that can be separated into:

```text
Floor_Number
Total_Floors
Floor_Ratio
Is_Ground
```

Likewise:

```text
Size / BHK
```

creates `Size_per_BHK`, which represents the approximate space available relative to the number of BHK units.

Additional transformations such as:

```text
Log_Size
Size_Squared
```

help expose alternative representations of property size that may capture nonlinear relationships.

---

## 🔍 Why RandomizedSearchCV?

Instead of manually selecting Random Forest parameters, **RandomizedSearchCV** evaluates sampled combinations from a predefined hyperparameter search space.

This provides:

- Systematic model tuning
- More efficient exploration than exhaustively testing every possible combination
- Cross-validation-based model selection
- Reduced dependence on manually chosen parameters

The final configuration obtained was:

```python
{
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None
}
```

---

## 🔄 Why 5-Fold Cross-Validation?

A single train/test split can provide a performance estimate that depends on one particular partition of the data.

With **5-fold cross-validation**, the model is repeatedly trained and validated across different subsets.

The project achieved:

```text
Cross-Validation R²: 0.6972 ± 0.0237
```

This provides additional evidence about the model's ability to generalize beyond a single split.

---

# 🤝 Contributing

Contributions that improve code quality, reproducibility, testing, documentation, or model analysis are welcome.

### Contribution Workflow

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/your-feature-name
```

3. Make your changes.

4. Run the test suite.

```bash
pytest tests/ -v
```

5. Commit the changes.

```bash
git add .
git commit -m "Describe your changes"
```

6. Push the branch.

```bash
git push origin feature/your-feature-name
```

7. Open a pull request describing the changes and their purpose.

Please keep contributions focused, documented, and consistent with the existing project structure.

---

# ⚡ Quick Reference

## Essential Commands

| Action | Command |
|---|---|
| Create virtual environment | `python -m venv venv` |
| Activate on Windows | `venv\Scripts\activate` |
| Activate on Linux/macOS | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Run ML pipeline | `python src/main.py` |
| Run tests | `pytest tests/ -v` |
| Run coverage | `pytest tests/ --cov=src --cov-report=html` |

---

## 📂 Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Main machine learning pipeline |
| `data/raw/House_Rent_Dataset.csv` | Raw house-rent dataset |
| `models/random_forest_baseline.pkl` | Saved baseline model |
| `models/random_forest_model_tuned.pkl` | Saved tuned Random Forest model |
| `sample/sample_unseen_houses.csv` | Unseen properties for inference |
| `sample/predictions_output_tuned.csv` | Predictions from tuned model |
| `tests/test_main.py` | Automated tests |
| `requirements.txt` | Python dependencies |
| `outputs/` | Evaluation results and visualizations |

---

# 🎯 Performance at a Glance

```text
Model                 Random Forest Regressor
R² Score              0.7619
MAE                   ₹8,647
RMSE                  ₹20,214
MAPE                  28.53%
Accuracy              91.55%
Cross-Validation R²   0.6972 ± 0.0237
Best CV R²            0.7902
Cross-Validation      5-Fold
Hyperparameter Search RandomizedSearchCV
Dataset Size          4,746 Records
Engineered Features   8+
```

---

# ⭐ Star This Project

If you find this project useful for learning about **machine learning, regression, feature engineering, hyperparameter tuning, model evaluation, or end-to-end ML pipelines**, consider starring the repository.

⭐ **GitHub Repository:** [github.com/23f3001167/house-rent-prediction](https://github.com/23f3001167/house-rent-prediction)

A star helps make the project easier for others interested in practical machine learning to discover.