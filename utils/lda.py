import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class my_lda:
  def __init__(self, n_components):
    self.n_components = n_components
    self.lda_components = None

  def fit(self, X, y):
    n_features = X.shape[1]
    mean_overall = np.mean(X, axis=0)

    #Macierze rozproszenia wewnątrz (S_W) i międzyklasowego (S_B)
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    n_classes = len(np.unique(y))
    components = min(self.n_components, n_classes - 1)

    for i in np.unique(y):
        X_i = X[y == i]
        mean_i = np.mean(X_i, axis=0)
        
        #S_W dla klasy i
        S_W += np.cov(X_i.T)
        
        #S_B dla klasy i
        n_i = X_i.shape[0]
        mean_diff = (mean_i - mean_overall).reshape(-1, 1)
        S_B += n_i * np.dot(mean_diff, mean_diff.T)

    #Rozwiązanie problemu własnego dla (S_W^-1) * S_B
    A = np.dot(np.linalg.inv(S_W), S_B)

    eigenvalues_lda, eigenvectors_lda = np.linalg.eig(A)

    eigen_pairs = [(np.abs(eigenvalues_lda[i]), eigenvectors_lda[:,i]) for i in range(len(eigenvalues_lda))]
    eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0], reverse=True)
    
    columns_to_stack = [eigen_pairs[i][1].reshape(-1, 1) for i in range(components)]

    self.lda_components = np.hstack(columns_to_stack).real
    X_lda = np.dot(X, self.lda_components)
    df_lda = pd.DataFrame(X_lda, columns=['LDA_1', 'LDA_2'])
    return X_lda, df_lda
