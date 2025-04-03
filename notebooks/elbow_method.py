#Three lines to make our compiler able to draw:
from main import read_preprocessed_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = read_preprocessed_data().select_dtypes(include=['int', 'float']).dropna()
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
kmeans = KMeans(n_clusters=6)
kmeans.fit(data)

plt.show()

