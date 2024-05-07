from abc import abstractmethod
from typing import List, Optional

import numpy as np

NEGATIVE_SAMPLER = "sampler"


class INegativeSampler:

    def __init__(self, n_samples: int, seed: Optional[int] = 42):

        self._n_samples = n_samples
        self._seed = seed

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def random(self) -> np.random.RandomState:
        return np.random.RandomState(self._seed)

    @abstractmethod
    def sample(self, sources: List, targets: List) -> List[List[str]]:
        pass
