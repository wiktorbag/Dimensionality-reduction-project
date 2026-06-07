import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import umap  

def isomap(X, n_components=2, n_neighbors=15, scale=True):
    """Redukcja wymiarowości z Isomap"""
    X = np.asarray(X)
    if scale:
        X = StandardScaler().fit_transform(X)
    return Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)

def umap_reduce(X, n_components=2, n_neighbors=15, min_dist=0.1, scale=True, random_state=1):
    """Redukcja wymiarowości z UMAP"""
    X = np.asarray(X)
    if scale:
        X = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)
