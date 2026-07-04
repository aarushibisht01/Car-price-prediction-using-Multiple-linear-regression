# Car Price Prediction Using Multiple Linear Regression

## Project Overview

This project predicts the selling price of a car using **Multiple Linear Regression**. It demonstrates a complete Machine Learning workflow, including data exploration, preprocessing, feature engineering, model training, evaluation, and visualization using Python and Scikit-learn.

The primary objective of this project is to understand how numerical and categorical features influence car prices while implementing a regression model from scratch.

## 📂 Dataset

* **Dataset:** Car Price Prediction Dataset
* **Source:** Kaggle
* **Records:** 2,500
* **Target Variable:** `Price`

### Features

* Car ID
* Brand
* Year
* Engine Size
* Fuel Type
* Transmission
* Mileage
* Condition
* Model
* Price (Target)

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

# Exploratory Data Analysis (EDA)

The following analyses were performed:

* Displayed dataset information
* Checked data types
* Statistical summary of numerical and categorical columns
* Checked missing values
* Checked duplicate records
* Counted unique values
* Brand distribution
* Fuel Type distribution
* Transmission distribution
* Condition distribution
* Model distribution
* Target variable (Price) distribution
* Numerical feature distributions using boxplots
* Categorical feature distributions using count plots
* Bivariate analysis
* Correlation heatmap

# Data Cleaning

* Removed duplicate records
* Removed missing values
* Dropped `Car ID`
* Created a new `Age` feature from the `Year` column
* Removed the original `Year` column

# Data Preprocessing

* Separated features and target variable
* Applied One-Hot Encoding to categorical features
* Used `drop="first"` to avoid the Dummy Variable Trap
* Split the dataset into training and testing sets
* Applied Standard Scaling to numerical features

# Model

**Algorithm Used**

* Multiple Linear Regression

# Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Training R² Score
* Testing R² Score

# Visualizations

The project includes the following visualizations:

* Price Distribution Histogram
* Numerical Feature Boxplots
* Categorical Feature Count Plots
* Mileage vs Price Scatter Plot
* Engine Size vs Price Scatter Plot
* Transmission vs Price Boxplot
* Fuel Type vs Price Boxplot
* Correlation Heatmap
* Actual vs Predicted Prices
* Residual Plot

# Results

The Multiple Linear Regression model was successfully trained and evaluated on the dataset.

Although the model achieved a relatively low R² score on this dataset, the project demonstrates the complete end-to-end machine learning pipeline, including data preprocessing, feature engineering, visualization, model training, and evaluation. It also highlights the importance of dataset quality and feature-target relationships when building predictive models.

# Learning Outcomes

Through this project, I learned:

* Performing Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Feature engineering
* One-Hot Encoding
* Feature scaling
* Building Multiple Linear Regression models
* Evaluating regression models using MAE, MSE, RMSE, and R²
* Visualizing data and model performance
* Interpreting regression results

## 👩‍💻 Author

**Aarushi Bisht**

B.Tech Computer Science Engineering (AI & ML)

Passionate about Artificial Intelligence, Machine Learning, and building practical data science projects.

⭐ If you found this project helpful, consider giving the repository a star!
