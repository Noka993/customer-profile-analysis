import pandas as pd
import matplotlib.pyplot as plt
from data import read_preprocessed_data


def plot_histograms(df, color_palette):
    """Plot histograms for non-binary columns."""
    nonbin_columns = [col for col in df.columns if df[col].nunique() > 3]
    axes = df[nonbin_columns].hist(
        figsize=(12, 10),
        layout=(4, 4),
        color=color_palette[0],
        edgecolor="black",
        grid=False,
    )

    for ax in axes.flatten():
        ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    plt.show()
    return nonbin_columns


def plot_pie_charts(df, color_palette, binary_columns):
    """Plot pie charts for binary columns with proper type handling."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.flatten()

    for i, col in enumerate(binary_columns):
        if i >= len(axes):
            break

        counts = df[col].value_counts()
        counts.index = counts.index.astype(str)

        counts.plot(
            kind="pie",
            ax=axes[i],
            colors=color_palette,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
            title=col,
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_boxplots(df, color_palette, columns):
    """Plot boxplots for specified columns."""
    ax = df[columns].plot(
        kind="box",
        subplots=True,
        figsize=(12, 10),
        layout=(5, 5),
        color=color_palette[1],
    )
    plt.tight_layout()
    plt.show()


def plot_bar_charts(df, color_palette, binary_columns):
    """Plot bar charts for binary columns."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.flatten()

    for i, col in enumerate(binary_columns):
        if i >= len(axes):
            break

        counts = df[col].value_counts()
        counts.index = counts.index.astype(str)
        counts.plot(
            kind="bar", ax=axes[i], color=color_palette, edgecolor="black", title=col
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def get_binary_columns(df):
    """Identify binary columns in the dataframe."""
    return [col for col in df.columns if df[col].nunique() <= 3]


def main():
    # Load and prepare data
    df = read_preprocessed_data()
    color_palette = ["#3a86ff", "#ff006e", "#8338ec"]

    # Plot visualizations
    nonbin_columns = plot_histograms(df, color_palette)

    binary_columns = get_binary_columns(df)
    print("Binary columns:", binary_columns)

    plot_pie_charts(df, color_palette, binary_columns)
    plot_boxplots(df, color_palette, nonbin_columns)
    plot_bar_charts(df, color_palette, binary_columns)


if __name__ == "__main__":
    main()
