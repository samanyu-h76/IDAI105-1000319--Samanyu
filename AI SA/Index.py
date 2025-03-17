# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "amazon.csv"
df = pd.read_csv(file_path)

# Data Cleaning & Preprocessing
# Convert price columns to numeric
df['discounted_price'] = df['discounted_price'].replace('[\u20B9,]', '', regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace('[\u20B9,]', '', regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].replace(',', '', regex=True).astype(float)

# Drop missing values
df.dropna(inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 5))
sns.histplot(df['discounted_price'], bins=50, kde=True)
plt.title("Distribution of Discounted Prices")
plt.show()

plt.figure(figsize=(12, 5))
sns.scatterplot(x=df['actual_price'], y=df['discounted_price'])
plt.title("Actual Price vs Discounted Price")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Customer Segmentation (Clustering)
features = df[['discounted_price', 'actual_price', 'rating', 'rating_count']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['customer_segment'], palette='viridis')
plt.title("Customer Segmentation using K-Means")
plt.show()

# Association Rule Mining
basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().fillna(0)
basket = basket.map(lambda x: True if x > 0 else False)

frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering min_support.")
else:
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    print("Top Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# User Behavior Analysis
review_lengths = df['review_content'].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 5))
sns.histplot(review_lengths, bins=30, kde=True)
plt.title("Distribution of Review Lengths")
plt.show()

print("Average Rating by Category:")
print(df.groupby('category')['rating'].mean().sort_values(ascending=False))
