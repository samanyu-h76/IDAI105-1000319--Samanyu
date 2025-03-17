import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler

# Streamlit UI Setup
st.title("Amazon Dataset Analysis")

# Load dataset from GitHub
file_path = "amazon.csv"
df = pd.read_csv(file_path)

# Data Cleaning
df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
df['discount_percentage'] = df['discount_percentage'].replace('%', '', regex=True).astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].replace(',', '', regex=True).astype(float)
df.dropna(inplace=True)

# Display Data Overview
st.subheader("Dataset Overview")
st.write(df.head())

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")

# Discounted Price Distribution
st.write("### Distribution of Discounted Prices")
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(df['discounted_price'], bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Scatter Plot: Actual Price vs Discounted Price
st.write("### Actual Price vs Discounted Price")
fig, ax = plt.subplots(figsize=(12, 5))
sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Customer Segmentation using K-Means
st.subheader("Customer Segmentation")
features = df[['discounted_price', 'actual_price', 'rating', 'rating_count']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['customer_segment'], palette='viridis', ax=ax)
st.pyplot(fig)

# Association Rule Mining
st.subheader("Association Rule Mining")
basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().fillna(0)
basket = basket.map(lambda x: True if x > 0 else False)

frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
if frequent_itemsets.empty:
    st.write("No frequent itemsets found. Try lowering min_support.")
else:
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    st.write("### Top Association Rules")
    st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# User Behavior Analysis
st.subheader("User Behavior Analysis")
review_lengths = df['review_content'].apply(lambda x: len(str(x)))
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(review_lengths, bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Average Rating by Category
st.subheader("Average Rating by Category")
st.write(df.groupby('category')['rating'].mean().sort_values(ascending=False))
