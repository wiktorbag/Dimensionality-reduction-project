from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class Dataset:
    def __init__(self, X, y, name=None):
        self.X = X
        self.y = y
        self.name = name

    def plot_projection(self, X_proj, title=None):
        plt.figure(figsize=(8, 6))
        mapper = plt.cm.get_cmap('viridis', len(set(self.y)))
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=self.y, cmap=mapper, alpha=0.7)
        plt.title(title or self.name)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid()
        plt.show()
    


class DatasetEvaluator:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=42)
        self.unique_labels = set(self.dataset.y)
        self.reductions = None

    def scale(self, scaler):
        print(f"Scaling with {scaler.__class__.__name__}...")
        if scaler != None:
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
      

    def _evaluate_reduction_with_kmeans(self, X_train_reduced, X_test_reduced):
        kmeans = KMeans(n_clusters=len(self.unique_labels), random_state=42)
   

        train_clusters = kmeans.fit_predict(X_train_reduced)

        test_clusters = kmeans.predict(X_test_reduced)

        return {
            "train_clusters": train_clusters,
            "test_clusters": test_clusters,
            "train_NMI": normalized_mutual_info_score(self.y_train, train_clusters),
            "test_NMI": normalized_mutual_info_score(self.y_test, test_clusters),
            "train_ARI": adjusted_rand_score(self.y_train, train_clusters),
            "test_ARI": adjusted_rand_score(self.y_test, test_clusters),
            "train_Silhouette": silhouette_score(X_train_reduced, train_clusters),
            "test_Silhouette": silhouette_score(X_test_reduced, test_clusters),
            "train_Calinski-Harabasz": calinski_harabasz_score(X_train_reduced, train_clusters),
            "test_Calinski-Harabasz": calinski_harabasz_score(X_test_reduced, test_clusters),
        }
    
    def _evaluate_trustworthiness(self, X_reduced):
        return trustworthiness(self.X, X_reduced, n_neighbors=5)
    
    def _evaluate_knn_classification(self, X_train_reduced, X_test_reduced):
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_reduced, self.y_train)
        knn_predictions = knn.predict(X_test_reduced)
        return knn_predictions
    

    def predict_on_reduced_with_logistic_regression(self, X_train_reduced, X_test_reduced):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_reduced, self.y_train)
        lr_predictions = lr.predict(X_test_reduced)
        return lr_predictions
    
   
    def compare_reductions(self, reductions):
        """reductions = {
            "pca": {2: X_pca_2dim, 3: X_pca_3dim},
            "isomap": {2: X_isomap_2dim, 3: X_isomap_3dim},
            "umap": {2: X_umap_2dim, 3: X_umap_3dim},
            "autoencoder": {2: X_ae_2dim, 3: X_ae_3dim}
        }
        """
        self.reductions = reductions
        results = {}

        for method, dims in reductions.items():
            results[method] = {}
            for dim, X_reduced in dims.items():
                nmi, sil, cal = self._evaluate_reduction_with_kmeans(X_reduced)
                trust = self._evaluate_trustworthiness(X_reduced)
                results[method][dim] = {
                    "NMI": nmi,
                    "Silhouette": sil,
                    "Calinski-Harabasz": cal,
                    "Trustworthiness": trust
                }

        return results
    

    
        
        

