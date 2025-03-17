# Amazon Data Analysis - README

##  Project Title: **Amazon Customer Segmentation & Market Basket Analysis**

##  Project Scope
This project focuses on analyzing Amazon's e-commerce data to:
- Segment customers based on their buying behavior.
- Identify relationships between frequently purchased products.
- Gain insights from customer reviews and ratings.
- Deploy the analysis on a **Streamlit dashboard** for interactive data exploration.

---

##  **Key Preprocessing Steps**
1. **Data Cleaning & Formatting**
   - Converted prices and discount percentages to numeric values.
   - Removed missing values for accurate analysis.
   - Encoded categorical data for clustering and association rule mining.
2. **Exploratory Data Analysis (EDA)**
   - **Histograms** to visualize price distributions.
   - **Scatter plots** to analyze price relationships.
   - **Heatmaps** to study correlations between product attributes.

---

##  **Key Findings & Insights**
1. **Customer Segmentation (K-Means Clustering)**
   - Segmented customers based on product pricing, ratings, and purchase behavior.
   - Found distinct customer groups: Budget buyers, Premium buyers, and Discount hunters.
2. **Association Rule Mining (Apriori Algorithm)**
   - Identified frequently bought-together products.
   - Provided recommendations for **cross-selling and product bundling.**
3. **User Behavior Analysis**
   - Analyzed review lengths and ratings to understand customer feedback.
   - Identified top-rated and poorly-rated product categories.

---

##  **Streamlit Deployment & Functionality**
- The project is **deployed on Streamlit** for interactive exploration.
- Users can:
  - **View visualizations** (histograms, scatter plots, heatmaps).
  - **Interact with clustering results** using an intuitive dashboard.
  - **See association rule mining results** for product recommendations.

🔗 **Live Streamlit App Link**: https://idai105-1000319--samanyu-3nnq22ertrcga9cezprbda.streamlit.app/

---

## 📂 **Repository Structure**
```
📦 Amazon-Data-Analysis
 ┣ 📜 Index.py  # Main Streamlit App
 ┣ 📜 amazon.csv  # Dataset (If using local hosting)
 ┣ 📜 requirements.txt  # Dependencies
 ┗ 📜 README.md  # Project Documentation
```

---

## 📚 **References**
- [Market Basket Analysis Guide](https://yourselleragency.com/blog/market-basket-analysis-benefits-strategies)
- [K-Means Clustering](https://neptune.ai/blog/k-means-clustering)
- [Streamlit Documentation](https://docs.streamlit.io)

---

🔹 **Author**: Samanyu H
🔹 **Contact**: samanyu100h@gmail.com
🔹 **Date**: March 2025
