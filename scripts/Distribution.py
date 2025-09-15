"""
distribution.py
A program to analyze the distribution of an images dataset.
Given a root directory as input, it will:
- Count the number of images in each subdirectory (representing a category)
- Generate a pie chart and a bar chart showing the distribution per category
This script will work with datasets organized as follow:
    directory/
        subdirectory/
            image1.jpg
            image2.jpg
            ...
        subdirectory/
            ...
        ...

`python distribution.py /path/to/dataset`
"""

import argparse
import os
import sys

import pandas as pd

# import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from utils.logger import get_logger

PLOT_OUTPUT_DIR = "plots"

logger = get_logger(__name__)

def plot_multiple_distributions(plot_title:str, distributions: list[tuple[pd.DataFrame, str]]) -> None:
    """
    Generates and displays multiple combined pie+bar charts, one per dataset.
    Up to 5 datasets. Displays inline (does not save images).

    Args:
        distributions: List of (DataFrame, title) tuples.
                       Each DataFrame must have 'category' and 'count' columns.
    """
    if not distributions:
        raise ValueError("No distributions provided.")
    if len(distributions) > 5:
        raise ValueError("Too many datasets to plot. Max allowed: 5.")

    n = len(distributions)
    fig = make_subplots(
        rows=n,
        cols=2,
        subplot_titles=[
            f"{title} - Pie Chart" if i % 2 == 0 else f"{title} - Bar Chart"
            for df, title in distributions
            for i in range(2)
        ],
        specs=[[{"type": "domain"}, {"type": "xy"}] for _ in range(n)],
        vertical_spacing=0.15,
    )

    for row_idx, (df, title) in enumerate(distributions, start=1):
        if df.empty or not {"category", "count"}.issubset(df.columns):
            raise ValueError(f"Invalid DataFrame for dataset '{title}'")

        categories = df["category"].tolist()
        color_map = {
            cat: DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
            for i, cat in enumerate(categories)
        }

        fig.add_trace(
            go.Pie(
                labels=df["category"],
                values=df["count"],
                marker=dict(colors=[color_map[cat] for cat in df["category"]]),
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df["category"],
                y=df["count"],
                marker_color=[color_map[cat] for cat in df["category"]],
            ),
            row=row_idx,
            col=2,
        )

    fig.update_layout(
        height=500 * n,
        width=1400,
        title_text="Dataset Distributions",
        showlegend=False,
    )

    output_path = os.path.join(PLOT_OUTPUT_DIR, f"{plot_title}_multiple_chart.png")
    fig.write_image(output_path, width=1000, height=1500)

    logger.info(f"Distribution chart saved to: {output_path}")


def plot_distribution_combined(df: pd.DataFrame, title: str) -> None:
    """
    Generates combined pie and bar charts from df, saves it as an image.
    """
    if df.empty or not {"category", "count"}.issubset(df.columns):
        raise ValueError(
            "Invalid DataFrame: must include 'category' and 'count' columns."
        )

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    categories = df["category"].tolist()
    color_map = {
        cat: DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
        for i, cat in enumerate(categories)
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{title} - Pie Chart", f"{title} - Bar Chart"),
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )

    fig.add_trace(
        go.Pie(
            name="Pie",
            labels=df["category"],
            values=df["count"],
            marker=dict(colors=[color_map[cat] for cat in df["category"]]),
        ),
        row=1,
        col=1,
    )

    for i, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["category"]],
                y=[row["count"]],
                name=row["category"],
                marker_color=color_map[row["category"]],
            ),
            row=1,
            col=2,
        )

    # fig.add_trace(
    #     go.Bar(
    #         name="Bar",
    #         x=df["category"], y=df["count"]),
    #     row=1, col=2
    # )

    fig.update_layout(
        title_text=f"Dataset '{title}' Distribution",
        showlegend=False,
        height=800,
        width=1500,
    )
    fig.update_xaxes(tickangle=0, title_text="Category", row=1, col=2)
    fig.update_yaxes(
        tickmode="linear",
        tick0=0,
        dtick=200,
        title_text="Number of Images",
        row=1,
        col=2,
    )

    output_path = os.path.join(PLOT_OUTPUT_DIR, f"{title}_combined_chart.png")
    fig.write_image(output_path, width=1000, height=500)

    logger.info(f"Combined distribution chart saved to: {output_path}")


# def plot_distribution(df: pd.DataFrame, title: str) -> None:
#     """
#     Generates pie and bar charts from a DataFrame, saves them as images
#     """
#     if df.empty or not {"category", "count"}.issubset(df.columns):
#         raise ValueError("Invalid DataFrame.")

#     os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

#     fig_pie = px.pie(
#         df,
#         names="category",
#         values="count",
#         title=f"Dataset '{title}' Distribution - Pie Chart",
#     )

#     fig_bar = px.bar(
#         df,
#         x="category",
#         y="count",
#         color="category",
#         title=f"Dataset '{title}' Distribution - Bar Chart",
#         labels={"category": "Category", "count": "Number of Images"},
#     )
#     fig_bar.update_layout(
#         showlegend=False,
#         xaxis_tickangle=-45,
#         yaxis=dict(tickmode="linear", tick0=0, dtick=200),
#     )

#     output_dir = "plots"
#     os.makedirs(output_dir, exist_ok=True)
#     pie_path = os.path.join(output_dir, f"{title}_pie_chart.png")
#     bar_path = os.path.join(output_dir, f"{title}_bar_chart.png")
#     fig_pie.write_image(pie_path)
#     fig_bar.write_image(bar_path)

#     logger.info(f"Distribution charts saved to: {pie_path}, {bar_path}")


def load_image_counts(directory_path) -> pd.DataFrame:
    """Browses the subdirectories and returns a df with category counts"""
    subdirs = os.listdir(directory_path)
    image_counts = {}
    for subdir in subdirs:
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            files = [
                f
                for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f))
            ]
            image_counts[subdir] = len(files)

    df = pd.DataFrame([{"category": k, "count": v} for k, v in image_counts.items()])
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    return df


def parse_args() -> argparse.Namespace:
    """Uses `argparse` to handle the argument : `directory`"""
    parser = argparse.ArgumentParser(
        description="A program to analyze the distribution of an images dataset."
    )
    parser.add_argument("directory", help="Path to the root image dataset directory")
    return parser.parse_args()


def main():
    try:
        logger.info("Starting Distribution.py program")
        args = parse_args()
        directory_path = args.directory

        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"'{directory_path}' is not a valid directory.")

        logger.info(f"Processing dataset at directory: {directory_path}")

        df = load_image_counts(directory_path)
        title = os.path.split(directory_path)[-1]

        logger.info(
            f"Dataset '{title}' "
            f"contains {df['category'].nunique()} categories "
            f"and {df['count'].sum()} images in total."
        )
        logger.info(df.sort_values("count", ascending=False))

        plot_distribution_combined(df=df, title=title)

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
