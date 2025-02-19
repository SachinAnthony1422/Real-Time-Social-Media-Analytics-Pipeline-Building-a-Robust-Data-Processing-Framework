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

# Step 8: Sentiment Analysis using VADER
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Caption'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
)

# Step 9: Hashtag Clustering using TF-IDF and K-Means
df['Hashtags'] = df['Hashtags'].fillna("NoHashtag")
df_filtered = df[df['Hashtags'] != "NoHashtag"]
if df_filtered['Hashtags'].nunique() > 1:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_filtered['Hashtags'])
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    print(f"Silhouette Score for Hashtag Clustering: {silhouette_avg:.4f}")
    df_filtered['hashtag_cluster'] = clusters
else:
    print("❌ Not enough valid hashtags for clustering.")
    df_filtered['hashtag_cluster'] = -1

df = df.merge(df_filtered[['Hashtags', 'hashtag_cluster']], on='Hashtags', how='left')

# Step 10: Engagement Prediction using Random Forest Regression
X = df[['Likes', 'Shares', 'Comments']]
y = df['Impressions']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Engagement Prediction Model MSE: {mse:.4f}")

# Step 11: Save trained model
joblib.dump(rf, "engagement_model.pkl")


