from data import (
    perform_gmm_clustering,
    read_preprocessed_data,
    apply_pca,
    summarize_clusters,
    plot_cluster_profiles,
    plot_clusters,
    print_pca_variance
)

color_palette = ["#70d6ff", "#ff70a6", "#ff9770", "#ffd670"]

n_clusters = [3, 4]

for n in n_clusters:
    # Read the preprocessed data
    data_scaled = read_preprocessed_data(std=False, robust=True, gmm=True, le=False, he=True)

    # Perform GMM clustering
    data_scaled, gmm_model = perform_gmm_clustering(n_clusters=n, data_scaled=data_scaled)

    data_scaled, pca = apply_pca(data_scaled)

    plot_clusters(data_scaled, color_palette, title="Clusters (PCA)")
    
    # Print explained variance for PCA components
    print_pca_variance(pca)  # Display explained variance

    original_data = read_preprocessed_data(std=False, le=False)

    # Wypisujemy podsumowanie klastrów na konsolę
    summary = summarize_clusters(original_data, data_scaled)
    plot_cluster_profiles(summary, color_palette)
