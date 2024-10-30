# Wild Blueberry Yield Prediction - Kaggle Competition

![Leaderboard Rank](https://img.shields.io/badge/Leaderboard%20Rank-Top%2012%25-brightgreen)
![MAE](https://img.shields.io/badge/MAE-255.22-blue)

## 📘 Competition Summary
This repository contains my solution for a Kaggle competition aimed at predicting wild blueberry yield. Competing as a solo participant, I achieved a Mean Absolute Error (MAE) of **255.22**, securing a spot in the top **12%** of 180 teams.

## 🏆 Challenge Overview
The objective was to develop a regression model that could accurately forecast blueberry yields using classical ML techniques. Deep learning models were not allowed, with an emphasis on leveraging ensemble methods and traditional ML models.

## 📊 Dataset Description
The dataset is based on the **Wild Blueberry Pollination Simulation Model**, a validated simulation tool backed by over 30 years of observational data from Maine, USA, and Canadian Maritimes. The features capture various factors affecting pollination efficiency and yield, including:

- **Clonesize**: Average blueberry clone size (m²)
- **Honeybee, Bumblebee, Andrena, Osmia**: Densities of various bee species (bees/m²/min)
- **Temperature Features**: Maximum, minimum, and average daily air temperatures during blooming season (°C)
- **Rainfall Features**: Number of rainy days and average raining days during blooming season

The **train** and **test** sets were provided, with leaderboard rankings based on Mean Absolute Error (MAE).

## 🧩 Solution Approach

Given the classical ML constraint, my approach included:

1. **Exploratory Data Analysis (EDA)**: Investigated feature distributions, correlations, and applied feature engineering.
2. **Feature Selection**: Selected features based on correlations and importance to improve predictive accuracy.
3. **Model Selection**: Tested several classical ML models:
   - **Linear Regression**
   - **Random Forest Regressor**
   - **Gradient Boosting Regressor**
4. **Ensembling**: Combined models using ensemble methods to enhance performance.
5. **Hyperparameter Tuning**: Applied grid search and cross-validation to fine-tune the parameters, optimizing for MAE.

## 🚀 Results
- **MAE**: 255.22
- **Leaderboard Position**: Top 12% out of 180 teams

## 🔧 Libraries and Tools
- Python 3.x
- Key Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## 📂 Repository Structure
```plaintext
├── notebooks/               # Jupyter notebooks for EDA and modeling
├── src/                     # Source code for preprocessing and model training
├── data/                    # Placeholder for dataset (not included due to Kaggle terms)
├── README.md                # Project overview
