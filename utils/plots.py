import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np


def plot_2d_projection(X, y, title=None, ax=None, show_colorbar=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="viridis",
        alpha=0.7

    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(*scatter.legend_elements(), title="Classes")
    ax.grid(True)
    if show_colorbar:
        fig.colorbar(scatter, ax=ax, label="Class Label")
    return fig, ax

def plot_2d_projection_comparison(projections, y, title="Projection Comparison"):
    fig, axes = plt.subplots(2, len(projections)//2+1, figsize=(12, 8))
    for projection, ax in zip(projections, axes.flatten()):
        plot_2d_projection(projection[0], y, ax=ax, show_colorbar=False, title=projection[1])
    
    plt.tight_layout()

    plt.show()


def plot_3d_interactive(X_3d, y=None, title="3D embedding"):
    df_plot = pd.DataFrame({
        "x": X_3d[:, 0],
        "y": X_3d[:, 1],
        "z": X_3d[:, 2],
    })

    if y is not None:
        df_plot["label"] = y.astype(str)
        fig = px.scatter_3d(
            df_plot,
            x="x",
            y="y",
            z="z",
            color="label",
            title=title,
            opacity=0.75
        )

    else:
        fig = px.scatter_3d(
            df_plot,
            x="x",
            y="y",
            z="z",
            title=title,
            opacity=0.75
        )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        width=700,
        height=500,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        )
    )

    fig.show()



def plot_projection_comparison(projections, y, title="Projection Comparison"):
    plt.figure(figsize=(15, 5))
    for i, (name, X_proj) in enumerate(projections.items()):
        plt.subplot(1, len(projections), i + 1)
        scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Class Label')
        plt.title(name)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def radar_plot(values, labels, title=None):
    values = np.asarray(values)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values = np.concatenate([values, [values[0]]])
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(True)
    plt.show()


