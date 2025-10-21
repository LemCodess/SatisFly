# SatisFly: Predicting Airline Passenger Satisfaction with Machine Learning

**SatisFly** is a machine learning-based project designed to predict airline passenger satisfaction using demographic, flight, and service-related features. The project investigates key factors influencing satisfaction and applies classification models to accurately distinguish between _satisfied_ and _neutral/dissatisfied_ passengers.

---

## Dataset Summary

- **Source**: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data
- **Total Records**: 129,880
  - Training Set: 103,904
  - Testing Set: 25,976
- **Total Features**: 25
  - Categorical: `Gender`, `Customer Type`, `Type of Travel`, `Class`, `satisfaction`
  - Numerical: e.g. `Age`, `Flight Distance`, `Inflight wifi service`, `Cleanliness`, etc.
- **Target Variable**: `satisfaction`
  - Values: `0` (Neutral or Dissatisfied), `1` (Satisfied)

---

## 🔍 Project Steps

### 1. Exploratory Data Analysis (EDA)

- Distribution analysis of target and categorical/numerical features
- Correlation heatmap revealed strong influence of:
  - `Online boarding`
  - `Inflight entertainment`
  - `Type of Travel`

### 2. Data Preprocessing

- **Missing Values**: `Arrival Delay in Minutes` filled with `0`
- **Encoding**: Label encoding for categorical features
- **Feature Scaling**: Standardization using `StandardScaler`
- **Feature Reduction**: Dropped irrelevant features (`Gate location`, etc.) based on correlation and T-tests

### 3. Principal Component Analysis (PCA)

- PCA used to analyze variance and feature importance
- Not used for dimensionality reduction, only for insight
- Most frequent feature across top PCs: `Gate location` (though later dropped)

### 4. Inferential Statistics

- **T-tests** performed on features to measure significance with respect to satisfaction
- Significant features (p < 0.05):
  - `Departure Delay`, `Arrival Delay`, `Time Convenience`

### 5. Models Used

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| Random Forest | 96.17%   | 96.20%    | 96.17% | 96.17%   |
| Decision Tree | 94.67%   | 94.67%    | 94.67% | 94.67%   |
| KNN           | 93.32%   | 93.39%    | 93.32% | 93.29%   |
| SVM           | 87.04%   | 87.03%    | 87.04% | 87.01%   |

> **Best Performer**: Random Forest Classifier

---

## Evaluation Metrics

- **Confusion Matrices**
- **Bar Charts** comparing:
  - Accuracy
  - Precision (per class)
  - Recall (per class)
  - F1 Score (per class)

---

## Dependencies

```bash
pip install -r requirements.txt
```
## How to run
**Option 1**
1. Open the notebook file (SatisFly.ipynb) in Jupyter Notebook or VS Code.

2. Run all cells sequentially to execute data preprocessing, model training, and evaluation.

3. View the output visualizations (correlation heatmap, PCA plots, confusion matrices, etc.) directly in the notebook.

**Option 2**
Run the Streamlit app

```bash
streamlit run app.py
```
## Key Insights

Online services (Online boarding, Inflight entertainment) are most correlated with satisfaction.

Delay times and travel types significantly affect satisfaction.

Random Forest outperformed other models across all metrics.

PCA and statistical testing helped validate feature importance.
