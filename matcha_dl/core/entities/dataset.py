

import numpy as np
import pandas as pd
from typing import Optional

from matcha_dl.core.entities.dp.anchor_mappings import AnchoredOntoMappings

DataFrame = pd.DataFrame

class MlpDataset:

    def __init__(self, dataframe: DataFrame, ref: Optional[DataFrame] = None, candidates: Optional[AnchoredOntoMappings] = None):

        self._ref = ref
        self._df = dataframe
        self._candidates = candidates

    @property
    def reference(self):
        return self._ref
    
    @property
    def candidates(self):
        return self._candidates

    @property
    def dataframe(self):
        return self._df

    def x(self, kind: Optional[str] = 'train'):

        return np.array(self.dataframe[self._df[kind]]['Features'].values.tolist())

    def y(self, kind='train'):
        return self.dataframe[self.dataframe[kind]]['Labels'].values