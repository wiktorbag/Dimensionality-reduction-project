from dataclasses import dataclass
import matplotlib.pyplot as plt



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
        self.labels = dataset.y
        self.X = dataset.X
        self.reductions = None
        
    def evaluate_reduction(self, reduction_name, X_reduced):
        pass
        
        
        

