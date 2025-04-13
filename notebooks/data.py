import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
import os


def read_preprocessed_data(
    file_name="marketing_campaign.csv",
    std=True,
    minmax=False,
    robust=False,
    le=True,
    he=False,
    gmm=False,
    outliers=True,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    file_path = os.path.join(root_dir, "data/", file_name)
    print(file_path)
    df = pd.read_csv(file_path, sep="\t")

    # Brakujące dane
    df = df.dropna()

    # Bierzemy 2021, bo wtedy był ostatnio modyfikowany plik
    df["Age"] = 2021 - df["Year_Birth"]

    # Zamiana dat na ilość dni bycia klientem
    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    reference_date = pd.to_datetime("08-02-2021", format="%d-%m-%Y", errors="coerce")
    df["Dt_Customer"] = (reference_date - df["Dt_Customer"]).dt.days

    # Dodajemy sumę wydanych pieniędzy w okresie dwóch lat
    df["Spent"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    columns_to_scale = [
        "Age",
        "Income",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Dt_Customer",
        "Spent",
    ]
    if outliers:
        df = remove_outliers(df)

    # Ujednolicenie statusów cywilnych
    df["Marital_Status"] = df["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Absurd": "Alone",
            "Widow": "Alone",
            "YOLO": "Alone",
            "Divorced": "Alone",
            "Single": "Alone",
        }
    )

    # Ujednolicenie edukacji
    df["Education"] = df["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
    )

    if he:
        # Dane kategoryczne, one hot encoding
        s = df.dtypes == "object"
        object_cols = list(s[s].index)

        # Przekształcamy na dane numeryczne
        df = pd.get_dummies(df, columns=object_cols)

    if le:
        le = LabelEncoder()
        s = df.dtypes == "object"
        object_cols = list(s[s].index)

        for col in object_cols:
            df[col] = le.fit_transform(df[col])

    if std:
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    if minmax:
        scaler = MinMaxScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    if robust:
        scaler = RobustScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    del (
        df["Z_CostContact"],
        df["Z_Revenue"],
        df["ID"],
        df["Year_Birth"],
        df["Dt_Customer"],
    )
    if gmm:
        for col in df.columns:
            if len(df[col].unique()) < 9:
                df.drop(col, axis=1)
    return df


def outliers_statistics(df):
    outliers_count = []

    for col in df.columns:
        if df[col].dtypes in [float, int]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            outliers = df[(df[col] < (q1 - 1.5 * IQR)) | (df[col] > (q3 + 1.5 * IQR))]
            outliers_count.append(outliers.shape[0])
        else:
            outliers_count.append(0)

    outliers_percentage = []
    for value in outliers_count:
        outliers_percentage.append(float(value) / len(df))

    outliers_df = pd.DataFrame(
        [outliers_count, outliers_percentage], columns=df.columns
    )
    outliers_df.index = ["Ilość wartości skrajnych", "Procent wartości skrajnych"]

    return outliers_df


def remove_outliers(df2):
    outliers_percentages = outliers_statistics(df2)
    df = df2.copy()
    for i, col in enumerate(df.columns):
        if col in [
            "Income",
            "Recency",
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "Age",
            "Spent",
        ]:
            if df[col].dtypes in [float, int]:
                q1 = df2[col].quantile(0.25)
                q3 = df2[col].quantile(0.75)
                IQR = q3 - q1
                upper_bound = q3 + 1.5 * IQR
                lower_bound = q1 - 1.5 * IQR
                outliers_percentage = outliers_percentages.iloc[1, i]
                if outliers_percentage < 0.05:
                    df[col] = df[col].where(
                        ~((df[col] < lower_bound) | (df[col] > upper_bound))
                    )
                else:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    cleaned_df = df.dropna()

    return cleaned_df


def general_statistics(df):
    stat_cols = [col for col in df.columns if df[col].nunique() > 9]
    num_cols = pd.DataFrame(
        df, columns=stat_cols
    )  # df.select_dtypes(include=['int', 'float'])
    statystyki = {
        "Średnia": num_cols.mean(),
        "Mediana": num_cols.median(),
        "Minimum": num_cols.min(),
        "Maksimum": num_cols.max(),
        "Odchylenie Standardowe": num_cols.std(),
        "Skośność": num_cols.skew(),
    }
    return statystyki


def jaccard_res(df1, df2):
    # df1= arg1.copy()
    # df1['id'] = df1.index
    # df1 = pd.DataFrame(df1, columns=['id','Cluster'])
    #   df2= arg2.copy()
    #  df2['id'] = df2.index
    # df2 = pd.DataFrame(df2, columns=['id','Cluster'])
    # for i in range(len(df1["Clusters"].unique())):
    #   c1= = df1[df1['Clusters'] == i]
    #  c2 = df2[df2["Clusters"==i]]
    # overlap = pd.merge(c1, c2, how='inner')
    union = pd.concat([df1, df2]).drop_duplicates()
    overlap = pd.merge(df1, df2, how="inner").drop_duplicates()
    # union = pd.concat([df1, df2]).drop_duplicates()

    # if len(union) !=0 else 0
    return len(overlap) / len(union)


def optimal_jaccard(arg1, arg2, result=True):
    # assuming 4 clustesr
    jaccard_results = []
    jaccard_replacements = []
    for n in range(4):
        for l in range(3):
            for k in range(2):
                temp = ""
                df1 = arg1.copy()
                df1["id"] = df1.index
                df1 = pd.DataFrame(df1, columns=["Cluster", "id"])
                df2 = arg2.copy()
                df2["id"] = df2.index
                df2 = pd.DataFrame(df2, columns=["Cluster", "id"])
                df3 = df2.copy()
                vals = arg1["Cluster"].unique().tolist()
                vals.sort()
                #    df2.loc[df2["Cluster"] == 1,"Cluster"] = -101
                #   df2.loc[df2["Cluster"] == -101,"Cluster"] = 1

                #
                temp += "0: " + str(vals[n])
                df2.loc[df2["Cluster"] == vals[n], "Cluster"] = -100

                vals.pop(n)
                df2.loc[df2["Cluster"] == vals[l], "Cluster"] = -101
                temp += "   1: " + str(vals[l])
                vals.pop(l)

                df2.loc[df2["Cluster"] == vals[k], "Cluster"] = -102
                temp += "   2: " + str(vals[k])
                vals.pop(k)
                df2.loc[df2["Cluster"] == vals[0], "Cluster"] = -103
                temp += "   3: " + str(vals[0])
                # print(df2["Cluster"].unique())
                df2.loc[df2["Cluster"] == -100, "Cluster"] = 0
                df2.loc[df2["Cluster"] == -101, "Cluster"] = 1
                df2.loc[df2["Cluster"] == -102, "Cluster"] = 2
                df2.loc[df2["Cluster"] == -103, "Cluster"] = 3
                jaccard_results.append(jaccard_res(df1, df2))
                jaccard_replacements.append(temp)
            # print(df2["Cluster"].unique())
    # print(jaccard_results.index(max(jaccard_results)))
    #  print(jaccard_results)
    #    print(len(jaccard_results))
    if result:
        return max(jaccard_results)

    return jaccard_replacements[jaccard_results.index(max(jaccard_results))]


def apply_kmeans(data, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    data = data.copy()
    data["Cluster"] = model.fit_predict(data)
    return data, model


def apply_pca(data, n_components=2, drop_cluster=True):
    pca = PCA(n_components=n_components)
    features = data.drop("Cluster", axis=1) if drop_cluster else data
    components = pca.fit_transform(features)
    data = data.copy()
    for i in range(n_components):
        data[f"PCA{i + 1}"] = components[:, i]
    return data, pca


def plot_elbow(inertias, color="#70d6ff"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(inertias) + 1), inertias, marker="o", color=color)
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def plot_clusters(data, color_palette, title="Clusters visualized via PCA"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette=color_palette,
        s=60,
    )
    plt.title(title)
    plt.legend()
    plt.show()


def plot_clusters_with_centroids(data, pca, model, color_palette):
    centroids_pca = pca.transform(model.cluster_centers_)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data, x="PCA1", y="PCA2", hue="Cluster", palette=color_palette, s=60
    )
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="o",
        s=75,
        color="red",
        label="Centroids",
    )
    plt.title("KMeans Clusters Visualized via PCA")
    plt.legend()
    plt.show()


def summarize_clusters(original_data, clustered_data, display=True):
    original_data = original_data.copy()
    original_data["Cluster"] = clustered_data["Cluster"]

    summary = original_data.groupby("Cluster").mean(numeric_only=True)

    # Add mode for categorical variables
    object_cols = original_data.select_dtypes(include="object").columns
    for col in object_cols:
        summary[col + "_mode"] = original_data.groupby("Cluster")[col].agg(
            lambda x: x.mode()[0]
        )

    # Add counts
    summary["Count"] = original_data["Cluster"].value_counts().sort_index()

    if display:
        print(summary.transpose())

    return summary


def plot_cluster_profiles(summary, color_palette, cluster_labels=None):
    import matplotlib.patches as mpatches

    variables = ["Income", "Kidhome", "Teenhome", "Spent", "Age", "Count"]
    titles = [
        "Income ($)",
        "Kids at Home",
        "Teens at Home",
        "Total Spending ($)",
        "Average Age",
        "Customer Count",
    ]
    cluster_labels = summary.index if cluster_labels is None else cluster_labels

    # 🔄 Taller and less wide layout
    fig, axes = plt.subplots(3, 2, figsize=(7, 8))
    fig.suptitle("KMeans Cluster Profiles", fontsize=16, weight="bold")

    for ax, var, title in zip(axes.flatten(), variables, titles):
        values = summary[var]
        bars = ax.bar(cluster_labels, values, color=color_palette)
        ax.set_title(title, pad=10, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if var == "Count":
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        if var in ["Income", "Spent"]:
            ax.set_ylabel("Dollars ($)", fontsize=10)
        elif var == "Age":
            ax.set_ylabel("Years", fontsize=10)
        elif var == "Count":
            ax.set_ylabel("Customers", fontsize=10)
        else:
            ax.set_ylabel("Average", fontsize=10)

    # More space between plots
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    handles = [mpatches.Patch(color=color) for color in color_palette]
    labels = [f"Cluster {label}" for label in cluster_labels]

    # Move legend to bottom center if it fits better visually
    fig.legend(
        handles,
        labels,
        title="Clusters",
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        fontsize=10,
        title_fontsize=10,
    )

    plt.show()


def apply_tsne(data, color_palette, drop_cluster=True):
    """
    Apply t-SNE for dimensionality reduction and visualize the results.

    Parameters:
    - data: DataFrame with the data to apply t-SNE on
    - color_palette: Color palette to use for clusters
    - drop_cluster: Boolean to drop the 'Cluster' column if it exists before applying t-SNE

    Returns:
    - tsne_df: DataFrame containing the t-SNE components and Cluster labels
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features = data.drop("Cluster", axis=1) if drop_cluster else data
    tsne_components = tsne.fit_transform(features)

    # Create a new dataframe for visualization
    tsne_df = pd.DataFrame(
        {
            "TSNE1": tsne_components[:, 0],
            "TSNE2": tsne_components[:, 1],
            "Cluster": data["Cluster"],
        }
    )

    # Plot the results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=tsne_df,
        x="TSNE1",
        y="TSNE2",
        hue="Cluster",
        palette=color_palette,
        alpha=0.7,
        s=60,
    )
    plt.title("t-SNE Visualization of Customer Clusters", fontsize=14)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return tsne_df


def print_pca_variance(pca):
    """
    Print the explained variance ratio of the first two PCA components.

    Parameters:
    - pca: PCA object after fitting

    Prints:
    - Variance explained by PCA1 and PCA2
    - Total variance explained by the first two components
    """
    print("PCA1: ", round(pca.explained_variance_ratio_[0], 2))
    print("PCA2: ", round(pca.explained_variance_ratio_[1], 2))
    print("Suma wyjaśnionej wariancji: ", round(pca.explained_variance_ratio_.sum(), 2))


def perform_gmm_clustering(n_clusters, data_scaled):
    """
    Perform Gaussian Mixture Model clustering and return the data with clusters.
    """
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type="tied", random_state=42
    )
    data_scaled["Cluster"] = gmm.fit_predict(data_scaled)
    return data_scaled, gmm


if __name__ == "__main__":
    df = read_preprocessed_data()
    print(df.info())
