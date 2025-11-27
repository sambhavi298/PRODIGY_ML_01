# PRODIGY_ML_01
Implement a linear regression model to predict the prices of houses based on their square footage and no of bed rooms and bathrooms

# House Price Prediction using Linear Regression

Implement a linear regression model to predict the prices of houses based on their square footage and number of bedrooms and bathrooms.

## 📊 Dataset

Dataset used: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

## 📚 Libraries Used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR


## 🔍 Features

### Categorical Features
- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- prefarea
- furnishingstatus

### Numeric Features
- area
- bedrooms
- bathrooms
- stories
- parking

## 🛠️ Data Preprocessing

### Loading Data
df = pd.read_csv('Housing.csv')


### Exploratory Data Analysis
- Visualized categorical features using pie charts
- Created count plots for numeric features
- Generated histograms with percentage statistics
- Analyzed price distribution using box plots

### Outlier Detection and Removal
Used Z-score method with threshold r=3:
z_score = (df['area'] - df['area'].mean())/df['area'].std()
df = df[(z_score > (-1)*r) & (z_score < r)]


## 🤖 Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## 📈 Visualizations

The project includes:
- Scatter plots for area vs price correlation
- Pie charts for categorical feature distribution
- Count plots with bar labels
- Histograms with percentage statistics
- Box plots for price distribution analysis

## 📁 Project Structure

PRODIGY_ML_01/
├── README.md
└── Housing.csv

## 🚀 How to Run

1. Clone the repository
git clone https://github.com/sambhavi298/PRODIGY_ML_01.git

2. Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

3. Download the dataset from Kaggle
4. Run the Jupyter notebook/Python script
5. View model predictions and visualizations

## 📝 License

This project is open source and available under standard GitHub terms.

## 👤 Author

Created as part of Prodigy InfoTech internship task.
