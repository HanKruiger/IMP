# Samplers

Note on semantics: A __sampler__ samples point from the input dataset, while a __clusterer__ can generate new points in the same space as the input dataset.

* `N`: Number of sampler in input.
* `k`: Number of samples in output.
* `m`: Dimensionality (of both input and output).

## `RandomSampler`

This is the fastest sampler, that simply picks `k` random samples from the input dataset.

Pro's:

* Fast: `O(k)`
* Statistically has the same density distribution, as long as `k` not too small.

Con's:

* Undeterministic
* Can fail to preserve salient structures
