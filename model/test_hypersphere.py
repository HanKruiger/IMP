#!/usr/bin/env python3

import numpy as np
from hypersphere import HyperSphere

eps = np.finfo(np.float32).eps

def test_sampling():
    for radius in [2.3, 42, 0.001, 10]:
        for centroid in [np.array([-5, 2]), np.array([600, -3]), np.array([0.1, 80])]:
            hs = HyperSphere(centroid, radius)
            samples = hs.sample(100)
            assert(np.all(np.abs(np.linalg.norm(samples - centroid, axis=1) - radius) < eps))

def test_contains():
    hs = HyperSphere([3, 4], 1.5)
    
    assert([3, 4] in hs)

    hs2 = HyperSphere([3, 4.5], 0.9)
    hs3 = HyperSphere([3, 4.5], 1.1)
    assert(hs2 in hs)
    assert(hs3 not in hs)

    points = np.array([[4.25, 5.25], [3.75, 4.75], [3, 1], [2.5, 3]])
    assert(np.all(hs.contains(points) == np.array([False, True, False, True])))
    
    for radius in [2.3, 42, 0.001, 10]:
        for centroid in [np.array([-5, 2]), np.array([600, -3]), np.array([0.1, 80])]:
            hs = HyperSphere(centroid, radius)
            assert(np.all(hs.contains((hs.sample(100) - hs.centroid()) * 0.99 + hs.centroid())))
            assert(not np.any(hs.contains((hs.sample(100) - hs.centroid()) * 1.01 + hs.centroid())))

def sampling_density_experiments():
    from scipy.spatial.distance import pdist
    import matplotlib.pyplot as plt
    max_iters = 1000
    n_tries = 15

    n_dimss = np.arange(2, 61)
    samples_needed = np.zeros_like(n_dimss)

    for i, n_dims in enumerate(n_dimss):
        # Unit hypersphere around origin
        hs = HyperSphere(np.zeros(n_dims), 1)
        
        n_samples = 2
        iters = 0
        dense_enough = False
        while iters < max_iters:
            iters += 1

            dense_enough = True
            for _ in range(n_tries):
                samples = hs.sample(n_samples)
                pdists = pdist(samples)
                if pdists.min() > 1:
                    dense_enough = False
                    break

            if dense_enough:
                break
            n_samples += 2

        if dense_enough:
            print('{}d hypersphere dense enough with {} samples.'.format(n_dims, n_samples))
            samples_needed[i] = n_samples

    plt.figure(1)
    plt.semilogy(n_dimss, samples_needed)
    plt.xlabel('Dimensionality of unit hypersphere')
    plt.ylabel('Samples needed')
    plt.grid(True)

    n_dims = 6
    radii = np.linspace(1, 10, num=30)
    samples_needed = np.zeros_like(radii)
    for i, radius in enumerate(radii):
        hs = HyperSphere(np.zeros(n_dims), radius)

        n_samples = 2
        iters = 0
        dense_enough = False
        while iters < max_iters:
            iters += 1

            dense_enough = True
            for _ in range(n_tries):
                samples = hs.sample(n_samples)
                pdists = pdist(samples)
                if pdists.min() > 1:
                    dense_enough = False
                    break

            if dense_enough:
                break
            n_samples += 1

        if dense_enough:
            print('{}d hypersphere with radius {} dense enough with {} samples.'.format(n_dims, radius, n_samples))
            samples_needed[i] = n_samples

    plt.figure(2)
    plt.semilogy(radii, samples_needed)
    plt.xlabel('Radius of {}-d hypersphere'.format(n_dims))
    plt.ylabel('Samples needed')
    plt.grid(True)
    plt.show()

    assert(False)