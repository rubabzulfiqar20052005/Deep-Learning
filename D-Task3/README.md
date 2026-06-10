Conclusion

The performance of clustering algorithms depends on the shape of the dataset.

K-Means works best for spherical and well-separated clusters but struggles with non-spherical shapes such as moons and circles.

Agglomerative Clustering performs slightly better than K-Means on complex structures but may still fail to capture curved cluster boundaries.

DBSCAN successfully identifies arbitrarily shaped clusters and can detect noise points, making it more suitable for moon-shaped and circular datasets.

The K-Distance Graph was used to select an appropriate epsilon value for DBSCAN. Based on the visual results, DBSCAN produced the most meaningful clusters for complex datasets even when its Silhouette Score was not always the highest. Therefore, DBSCAN is generally preferred for non-spherical cluster structures.