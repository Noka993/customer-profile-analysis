from data import (
    read_preprocessed_data,
    apply_kmeans,
    apply_pca,
    plot_elbow,
    plot_clusters_with_centroids,
    summarize_clusters,
    plot_cluster_profiles,
    plot_clusters,
    print_pca_variance
)
from inertia import calculate_inertias

color_palette = ["#70d6ff", "#ff70a6", "#ff9770", "#ffd670"]

n_clusters = [3, 4]

for n in n_clusters:
    data_scaled = read_preprocessed_data(le=False, he=True)
    inertias = calculate_inertias(data_scaled)
    plot_elbow(inertias)

    data_scaled, model = apply_kmeans(data_scaled, n_clusters=n)
    data_scaled, pca = apply_pca(data_scaled)

    plot_clusters(data_scaled, color_palette, title="Clusters (PCA)")
    plot_clusters_with_centroids(data_scaled, pca, model, color_palette)
    
    # Print explained variance for PCA components
    print_pca_variance(pca)  # Display explained variance

    original_data = read_preprocessed_data(std=False, le=False)

    # Wypisujemy podsumowanie klastrów na konsolę
    summary = summarize_clusters(original_data, data_scaled)
    plot_cluster_profiles(summary, color_palette)
