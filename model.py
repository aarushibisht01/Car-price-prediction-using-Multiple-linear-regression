import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

#Target variable analysis for price distribution
plt.figure(figsize=(8,6))
sns.histplot(df["Price"],bins=30,kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()

#Distribution of numerical columns
numerical_columns=["Year", "Engine Size", "Mileage", "Price"]

for column in numerical_columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

#Distribution of categorical columns
categorical_columns=["Brand","Fuel Type","Transmission","Condition","Model"]

for column in categorical_columns:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df,x=column)
    plt.title(f"{column} Distribution")
    plt.xticks(rotation=45)
    plt.show()

#Bivariate analysis

list=["Mileage","Engine Size"]

for feature in list:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df,x=feature,y="Price")
    plt.title(f"{feature} vs Price")
    plt.show()

#price by transmission and Fuel type
list2=["Transmission","Fuel Type"]

for feature in list2:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df,x=feature,y="Price")
    plt.title(f"{feature} vs Price")
    plt.show()

#collinearity heatmap for multicollinearity check for numerical features
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Data cleaning

#Dropping the Car ID
df.drop("Car ID",axis=1,inplace=True)

df=df.dropna() #dropping features with missing values

#creating age of car column
present_year=2026
df["Age"]=present_year-df["Year"]
df.drop("Year",axis=1,inplace=True)

X=df.drop("Price",axis=1)
y=df["Price"]

#Encoding cateogrical values
transformers=[("encoder",OneHotEncoder(drop="first",sparse_output=False),
               ["Brand", "Fuel Type", "Transmission", "Condition", "Model"])]

preprocessor=ColumnTransformer(transformers,remainder="passthrough")

X=preprocessor.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
