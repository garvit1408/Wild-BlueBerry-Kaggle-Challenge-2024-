# Wild Blueberry Yield Prediction - Kaggle Competition

![Leaderboard Rank](https://img.shields.io/badge/Leaderboard%20Rank-Top%2012%25-brightgreen)
![MAE](https://img.shields.io/badge/MAE-255.22-blue)

## 📘 Competition Summary
This repository contains my solution for a Kaggle competition aimed at predicting wild blueberry yield. Competing as a solo participant, I achieved a Mean Absolute Error (MAE) of **255.22**, securing a spot in the top **12%** of 180 teams.

## 🏆 Challenge Overview
The objective was to develop a regression model to forecast blueberry yields using classical ML techniques, with a restriction against deep learning models.

## 📊 Dataset Description
The dataset is based on the **Wild Blueberry Pollination Simulation Model**, which captures how variables such as bee densities, temperatures, and rainfall influence yield. Key features include:

- **Clonesize**: Average blueberry clone size (m²)
- **Bee Density (Honeybee, Bumblebee, Andrena, Osmia)**: Densities of different bee species (bees/m²/min)
- **Temperature**: Maximum, minimum, and average daily air temperatures during blooming season (°C)
- **Rainfall**: Rainy days and average rainy days during blooming season

The **train** and **test** sets were provided, with leaderboard evaluation based on Mean Absolute Error (MAE).

## 🧩 Solution Approach

### 1. Data Preparation
- **Feature Selection**: Excluded non-predictive columns (`id`, `Row#`) and set the yield column as the target.
- **Data Split**: Used a 90-10 split for training and validation.
- **Scaling**: Standardized features using `StandardScaler`.

### 2. Model Selection
Chosen Model: **Gradient Boosting Regressor** due to its robustness in handling structured data and its feature importance capabilities. Model hyperparameters were fine-tuned based on validation performance.

### 3. Model Training and Evaluation
The model was initialized and trained with the following parameters:

```python
GradientBoostingRegressor(
    n_estimators=604,
    learning_rate=0.0321,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=5,
    subsample=0.8,
    loss='absolute_error',
    random_state=42
)
```

## 🚀 Results
- **MAE**: 255.22
- **Leaderboard Position**: Top 12% out of 180 teams

## 🔧 Libraries and Tools
- Python 3.x
- Key Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
