from torchvision import datasets, transforms
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Define the KMeans class
class KMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize centroids randomly
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in tqdm(range(self.max_iter)):
            # Assign each data point to the nearest centroid
            labels = self._assign_clusters(X)

            # Update centroids based on the mean of points in each cluster
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        return labels

    def _assign_clusters(self, X):
        # Compute distances between each data point and centroids
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))

        # Assign each data point to the nearest centroid
        return np.argmin(distances, axis=1)


def get_dataset(n_images=100):
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # Load MNIST dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Select first N images and labels
    data = mnist_trainset.data[:n_images].float().view(n_images, -1)
    targets = mnist_trainset.targets[:n_images]

    return data, targets


# Print each cluster's labels and display a sample image from each cluster
def plot_result(K, data, labels, targets):
    fig, axs = plt.subplots(3, 10, figsize=(15, 5))
    fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
    for i in range(K):
        cluster_indices = np.where(labels == i)[0]
        print(f"Cluster {i}: Labels: {targets[cluster_indices]}")

        # Select a random image from the cluster
        random_index = np.random.choice(cluster_indices)
        image = data[random_index].numpy().reshape(28, 28)
        representative_label = targets[random_index].item()

        # Plot the image
        ax = axs[i // 10, i % 10]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f"({i}) Label: {representative_label}")
        # Write prediction text below the picture plot
        ax.text(0.5, -0.15, f"Prediction: {targets[cluster_indices[0]].item()}",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def calculate_accuracy(K, labels, targets):
    # Calculate representative labels for each cluster
    representative_labels = []
    for i in range(K):
        cluster_indices = np.where(labels == i)[0]
        cluster_labels = targets[cluster_indices]
        most_common_label = Counter(cluster_labels.numpy()).most_common(1)[0][0]
        representative_labels.append(most_common_label)

    # Calculate accuracy
    correct_predictions = 0
    total_predictions = len(labels)

    for i in range(K):
        cluster_indices = np.where(labels == i)[0]
        cluster_labels = targets[cluster_indices]
        correct_predictions += np.sum(cluster_labels.numpy() == representative_labels[i])

    accuracy = correct_predictions / total_predictions
    print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")


def visualize_clusters(data, labels):
    # Perform dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

def evaluate_clustering(data, labels):
    # Compute silhouette score
    silhouette_avg = silhouette_score(data, labels)
    print(f'Silhouette Score: {silhouette_avg}')


def main():
    # Number of images
    N = 100
    # Number of clusters
    K = 30

    # Load Dataset
    data, targets = get_dataset(n_images=N)

    # Initialize KMeans object
    kmeans = KMeans(n_clusters=K, random_state=42)
    # Fit KMeans model
    labels = kmeans.fit(data.numpy())

    plot_result(K=K, data=data, labels=labels, targets=targets)
    calculate_accuracy(K=K, labels=labels, targets=targets)

    visualize_clusters(data, labels)
    evaluate_clustering(data, labels)


if __name__ == "__main__":
    main()