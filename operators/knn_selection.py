from sklearn.neighbors import NearestNeighbors

from model import Dataset

def knn_selection(source, n_samples, pos, sort=True):
    knn = NearestNeighbors(n_neighbors=n_samples)
    
    X = source.data()
    knn.fit(X)
    
    idcs_in_source = knn.kneighbors(pos.reshape(1, -1), return_distance=False).flatten()

    if sort:
        idcs_in_source.sort()

    data = source.data()[idcs_in_source, :]
    idcs_in_root = source.indices()[idcs_in_source]
    
    dataset = Dataset(data, idcs_in_root, name='KNN selection')
    
    return dataset
