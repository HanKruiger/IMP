import numpy as np

from model import Dataset
from operators import Operator


def random_sampling(source, n_samples, sort=True):
    idcs_in_source = np.random.choice(source.n_points(), n_samples, replace=False)

    if sort:
        idcs_in_source.sort()

    data = source.data()[idcs_in_source, :]
    indices = source.indices()[idcs_in_source]
    dataset = Dataset(data, indices, name='RND sampling')
    return dataset
