class my_pca:
  def __init__(self, n_components):
    self.n_components = n_components
    self.pca_components = None

  def fit(self, X):
    cov_matrix = np.cov(X, rowvar=False)
    
    eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(cov_matrix)

    idx_pca = np.argsort(eigenvalues_pca)[::-1]
    eigenvalues_pca = eigenvalues_pca[idx_pca]
    eigenvectors_pca = eigenvectors_pca[:, idx_pca]

    pca_components = eigenvectors_pca[:, :self.n_components]

    X_pca = np.dot(X, pca_components)
    df_pca = pd.DataFrame(X_pca, columns=['PC_1', 'PC_2'])
    return X_pca, df_pca
