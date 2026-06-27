# Customer_Segmentation
# Customer Segmentation using K-Means
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# لتحسين شكل الرسومات
sns.set(style="whitegrid")

# ==========================================
# 2. Load Dataset
# ==========================================

df = pd.read_csv(r"d:\my ai projects\2\Mall_Customers.csv")
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# 3. Dataset Information
# ==========================================

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# ==========================================
# 4. Exploratory Data Analysis (EDA)
# ==========================================

# Gender Distribution
plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Genre")
plt.title("Gender Distribution")
plt.show()

# Age Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, color="green")
plt.title("Annual Income Distribution")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, color="red")
plt.title("Spending Score Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 5. Feature Selection
# ==========================================

X = df[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# ==========================================
# 6. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# 7. Elbow Method
# ==========================================

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# ==========================================
# 8. Train KMeans
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ==========================================
# 9. Cluster Visualization
# ==========================================

plt.figure(figsize=(9,6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segments")
plt.show()

# ==========================================
# 10. Age vs Spending Score
# ==========================================

plt.figure(figsize=(9,6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set2",
    s=100
)

plt.title("Age vs Spending Score")
plt.show()

# ==========================================
# 11. Cluster Profile
# ==========================================

cluster_profile = df.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

print("\nCluster Profile")
print(cluster_profile)

# ==========================================
# 12. Cluster Size
# ==========================================

print("\nCustomers in Each Cluster")
print(df["Cluster"].value_counts())

# ==========================================
# 13. Save Results
# ==========================================

df.to_csv("customer_segments.csv", index=False)

print("\nFile Saved Successfully!")

# ==========================================
# 14. Marketing Recommendations
# ==========================================

print("\n==============================")
print("Marketing Report")
print("==============================")

for cluster in sorted(df["Cluster"].unique()):

    data = df[df["Cluster"] == cluster]

    age = data["Age"].mean()
    income = data["Annual Income (k$)"].mean()
    spend = data["Spending Score (1-100)"].mean()

    print(f"\nCluster {cluster}")

    print(f"Average Age: {age:.1f}")
    print(f"Average Income: {income:.1f}")
    print(f"Average Spending Score: {spend:.1f}")

    if spend >= 70:
        print("Recommendation: Premium Offers, Loyalty Programs, Exclusive Products")

    elif spend >= 50:
        print("Recommendation: Promotions and Membership Rewards")

    else:
        print("Recommendation: Discounts, Coupons, Re-engagement Campaigns")

print("\nProject Completed Successfully!")