import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
df = pd.read_csv("Titanic-Dataset.csv")

#checking for missing values
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)
print(df.isnull().sum())

#Convert categorical features into numerical using encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print(df.head())

#Normalize / Standardize the Numerical Features.
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

#Visualize outliers using boxplots and remove them.
# Boxplot for Age
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.show()

# Boxplot for Fare
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

#Remove outliers beyond ±3σ
df = df[(np.abs(df['Age']) < 3) & (np.abs(df['Fare']) < 3)]
print(df.shape)