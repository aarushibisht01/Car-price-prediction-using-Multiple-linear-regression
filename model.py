import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
df=pd.read_csv("dataset.csv")

#exploratory data analysis

print("First five rows of dataset:\n")
print(df.head())

print("Last five rows of dataset:\n")
print(df.tail())

print("Random 5 rows of dataset:\n")
print(df.sample(5))

print("Column names for dataset:\n")
print(df.columns)

print("Dataset information:\n")
print(df.info())

print("Datatypes of each columns:\n")
print(df.dtypes)

#statistical summary
print("Statistical summary for numerical columns:\n")
print(df.describe())

print("Statistical summary for categorical columns:\n")
print(df.describe(include='object'))

print("Finding missing values:\n")
print(df.isnull().sum())

print("Checking for duplicate rows:\n")
print(df.duplicated().sum())

print("Number of unique values in each column:\n")
print(print(df.nunique()))

print("Value counts for each categorical column:\n")

print("Brand distribution:\n")
print(df["Brand"].value_counts())

print("Fuel type distribution:\n")
print(df["Fuel Type"].value_counts())

print("Transmission distribution:\n")
print(df["Transmission"].value_counts())

print("Condition of car:\n")
print(df["Condition"].value_counts())

print("Model distribution:\n")
print(df["Model"].value_counts())