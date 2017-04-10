import numpy as np

from model import Dataset

def difference(d1, d2):
    idcs_1 = d1.indices()
    idcs_2 = d2.indices()

    assert(np.all(idcs_1 == np.sort(idcs_1)))

    diff = np.setdiff1d(idcs_1, idcs_2)
    idcs_in_d1 = np.searchsorted(idcs_1, diff)

    data = d1.data()[idcs_in_d1, :]
    indices = d1.indices()[idcs_in_d1]
    dataset = Dataset(data, indices, name='difference')

    return dataset