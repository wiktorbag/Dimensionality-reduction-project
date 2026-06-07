import matplotlib.pyplot as plt



def plot_2d_projection(X, y, title=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid()
    plt.show()


