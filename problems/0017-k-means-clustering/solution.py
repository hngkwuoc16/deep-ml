from scipy.spatial.distance import cdist
import numpy as np

def kmeans_assign_labels(points: list[tuple[float, ...]], centroids: list[tuple[float, ...]]) -> list[float]:
	D = cdist(points, centroids)
	labels = np.argmin(D, axis = 1)
	return labels

def kmeans_update_centroid(points: list[tuple[float, ...]], labels: list[float], k: int):
	points = np.array(points)
	new_centroids = np.zeros((k, len(points[0])))
	for i in range(k):
		same_label_points = points[labels == i, :]
		new_centroids[i, :] = np.mean(same_label_points, axis=0)
	return new_centroids

def has_converged(centroids, new_centroids):
	return (set([tuple(a) for a in centroids])) == (set([tuple(b) for b in new_centroids]))


def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	labels = kmeans_assign_labels(points, initial_centroids)
	new_centroids = kmeans_update_centroid(points, labels, k)
	while(has_converged(initial_centroids, new_centroids) == False and max_iterations > 0):
		initial_centroids = new_centroids
		labels = kmeans_assign_labels(points, initial_centroids)
		new_centroids = kmeans_update_centroid(points, labels, k)	
		max_iterations -= 1
	return [tuple(centroid) for centroid in new_centroids]