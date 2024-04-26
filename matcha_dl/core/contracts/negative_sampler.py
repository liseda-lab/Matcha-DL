from abc import abstractmethod

import numpy as np

NEGATIVE_SAMPLER = "sampler"

class INegativeSampler:

    def __init__(self, n_samples, seed=42):

        self._n_samples = n_samples
        self._seed = seed

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def random(self):
        return np.random.RandomState(self._seed)
    
    @abstractmethod
    def sample(self, sources, targets):
        pass