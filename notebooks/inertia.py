from sklearn.cluster import KMeans

def calculate_inertias(data_scaled):
    inertias = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)

    return inertias