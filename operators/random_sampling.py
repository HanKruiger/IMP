import numpy as np

from model import Dataset
from operators import Operator

from warnings import warn

# def random_sampling(source, n_samples, sort=True):
#     if n_samples < source.n_points():
#         idcs_in_source = np.random.choice(source.n_points(), min(n_samples, source.n_points()), replace=False)
#     else:
#         warn('Sample size larger than (or eq. to) source. Using all source samples.', RuntimeWarning)
#         idcs_in_source = np.arange(source.n_points())

#     if sort:
#         idcs_in_source.sort()

#     data = source.data()[idcs_in_source, :]
#     indices = source.indices()[idcs_in_source]
#     dataset = Dataset(data, indices, name='RND sampling')
#     return dataset
