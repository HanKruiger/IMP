import numpy as np
from model import Dataset


# TODO: Make this work on a list of datasets

def union(d1, d2, sort=True):
    idcs_1 = d1.indices()
    idcs_2 = d2.indices()

    assert(np.intersect1d(idcs_1, idcs_2).size == 0)

    data = np.concatenate([d1.data(), d2.data()], axis=0)
    indices = np.concatenate([idcs_1, idcs_2])

    if sort:
        order = np.argsort(indices)
        indices = indices[order]
        data = data[order, :]

    dataset = Dataset(data, indices, name='Union')

    return dataset
