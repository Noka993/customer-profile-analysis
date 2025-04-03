from main import read_preprocessed_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd

data = read_preprocessed_data()
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# print('Elbow Method to determine the number of clusters to be formed:')
# pca = PCA(n_components=3)
# pca.fit(data)
# PCA_ds = pd.DataFrame(pca.transform(data), columns=(["col1","col2", "col3"]))
# Elbow_M = KElbowVisualizer(KMeans(), k=10)
# Elbow_M.fit(PCA_ds)
# Elbow_M.show()

