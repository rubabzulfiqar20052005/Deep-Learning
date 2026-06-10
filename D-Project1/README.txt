FINAL PROJECT 1 – END TO END UNSUPERVISED LEARNING PIPELINE

Dataset:
Mall Customer Segmentation Dataset from Kaggle.

Objective:
The objective was to discover customer groups using unsupervised machine learning techniques.

EDA Findings:

1. No major missing values were found.
2. Customer income and spending score showed clear variation.
3. Gender distribution was relatively balanced.
4. Some natural grouping patterns were visible in scatter plots.

Preprocessing Steps:

1. Removed CustomerID column.
2. Converted Gender into numerical format.
3. Removed duplicate records.
4. Standardized all features using StandardScaler.
5. Applied PCA to reduce dimensionality to two principal components.

Clustering Algorithms Applied:

1. K-Means
2. Hierarchical Clustering
3. DBSCAN

Evaluation:
Silhouette Score was used to evaluate cluster quality.

Results:
K-Means achieved the highest Silhouette Score and produced clearly separated clusters.

Business Interpretation:
The clusters represent different customer segments such as:

* High Income High Spending Customers
* High Income Low Spending Customers
* Average Customers
* Budget Customers
* Young High-Spending Customers

Conclusion:
K-Means performed best because it generated compact and well-separated customer groups. These segments can be used for targeted marketing and customer relationship management.
