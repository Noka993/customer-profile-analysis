from data import (
    perform_gmm_clustering,
    read_preprocessed_data,
    apply_pca,
    summarize_clusters,
    plot_cluster_profiles,
    plot_clusters,
    print_pca_variance,
    outliers_statistics
)
import pandas as pd
import matplotlib.pyplot as plt
df = read_preprocessed_data(std=False, outliers=False)
print(outliers_statistics(df).to_string())
stat_cols = [col for col in df.columns if df[col].nunique()>9]




num_cols = pd.DataFrame(df,columns= stat_cols)
statystyki = {
'Średnia':num_cols.mean(),
'Mediana':num_cols.median(),
'Minimum':num_cols.min(),
'Maksimum':num_cols.max(),
'Odchylenie Standardowe':num_cols.std(),
'Skośność':num_cols.skew()
}

other_cols = [col for col in df.columns if col not in stat_cols]

print(other_cols)


color_pallette = [
    '#FFB3BA',  # Pastel Pink  
    '#FFDFBA',  # Pastel Orange  
    '#FFFFBA',  # Pastel Yellow  
    '#BAFFC9',  # Pastel Green  
    '#BAE1FF',  # Pastel Blue  
    '#D4A5FF',  # Pastel Purple  
    '#FFC3A0',  # Pastel Peach  
    '#A0E8FF',  # Pastel Sky Blue  
    '#B5EAD7',  # Pastel Mint  
    '#F8C8DC',  # Pastel Rose  
]


for stat in other_cols:
    ax = df[stat].value_counts().plot(
            kind="pie",
            subplots=True,
            figsize=(15, 200),
           # layout=(12, 1),
            colors=color_pallette,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90,
            ylabel="",
            legend=True,
            wedgeprops={'edgecolor': 'white'},
        title=stat



        #.astype(str)
        # .apply(df[stat].value_counts())
    )
    plt.show()

    plt.clf()
    for axis, col in zip(ax.flatten(), other_cols):
        for text in axis.texts:
            text.set_fontsize(30)
        axis.set_title(col, fontsize=30)




statystyki = pd.DataFrame(statystyki).transpose()

print(statystyki.to_string())