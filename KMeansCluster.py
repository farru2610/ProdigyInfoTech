import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

csv = "D:\Mall_Customers.csv"
data = pd.read_csv(csv)


num_clusters = 4

features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=num_clusters)
data['Cluster'] = kmeans.fit_predict(features)

plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering')
plt.show()
