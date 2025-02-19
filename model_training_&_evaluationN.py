# Step 1: Install necessary libraries
!pip install vaderSentiment  # Install the vaderSentiment library

# Step 2: Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import silhouette_score, mean_squared_error
import joblib  # Import joblib for saving the model

# Ignore warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Step 3: Load dataset
file_path = '/content/drive/MyDrive/Instagram_data_by_Bhanu.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Step 4: Data exploration
print(df.head())  # View first 5 rows
print(df.describe())  # Summary statistics
print(df.info())  # Dataset info
print(df.isna().sum())  # Check for missing values

# Step 5: Visualize missing values
msno.bar(df)
plt.show()

# Step 6: Histogram of numeric features
numeric_cols = df.select_dtypes(include=['number']).columns  # Select numeric columns only
plt.figure(figsize=(15, 10))
for plotnumber, column in enumerate(numeric_cols, 1):
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.histplot(df[column], kde=True)  # Histogram with density plot
        plt.xlabel(column)
plt.tight_layout()
plt.show()

# Step 7: Heatmap of correlation matrix
plt.figure(figsize=(20, 12))
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, linewidths=1, annot=True, fmt=".2f")
plt.show()

