import numpy as np

class HyperSphere:

    def __init__(self, centroid, radius=1):
        centroid = np.array(centroid)
        if centroid.ndim != 1:
            raise ValueError('centroid must be a 1-d array.')
        self._centroid = centroid
        self._radius = float(radius)

    def n_dims(self):
        return self.centroid().size

    def centroid(self):
        return self._centroid.copy()

    def radius(self):
        return self._radius

    def set_centroid(self, centroid):
        centroid = np.array(centroid)
        if centroid.ndim != 1:
            raise ValueError('centroid must be a 1-d array.')
        if self.centroid().size != centroid.size:
            raise ValueError('Changing the dimensionality of a HyperSphere is not allowed.')
        self._centroid = centroid

    def set_radius(self, radius):
        self._radius = float(radius)

    # Method by D.E. Knuth, The Art of Computer Programming, vol. 2
    def sample(self, n_samples=1):
        # Initialize samples from normal distribution.
        samples = np.random.normal(size=(n_samples, self.n_dims()))
        # Divide every sample by its norm
        samples /= np.linalg.norm(samples, axis=1)[:, np.newaxis]
        # Scale by the radius
        samples *= self.radius()
        # Move to centroid
        samples += self.centroid()
        if n_samples == 1:
            return samples.flatten()
        else:
            return samples

    def contains(self, points):
        points = np.atleast_2d(points)
        if points.ndim > 2:
            raise ValueError('points must be a 1-d or 2-d array.')
        if points.shape[1] != self.n_dims():
            raise ValueError('points must be of same dimensionality as HyperSphere.')
        # Move to origin
        points -= self.centroid()
        # Check if norms are small enough
        inside = np.linalg.norm(points, axis=1) <= self.radius()
        if points.shape[0] == 1:
            return inside[0]
        else:
            return inside
