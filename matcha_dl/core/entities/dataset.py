from typing import Optional

import numpy as np
import pandas as pd

from matcha_dl.core.entities.dp.anchor_mappings import AnchoredOntoMappings

DataFrame = pd.DataFrame


class MlpDataset:

    def __init__(
        self,
        dataframe: DataFrame,
        ref: Optional[DataFrame] = None,
        candidates: Optional[AnchoredOntoMappings] = None,
    ) -> None:

        self._ref = ref
        self._df = dataframe
        self._candidates = candidates

    @property
    def reference(self) -> DataFrame:
        return self._ref

    @property
    def candidates(self) -> AnchoredOntoMappings:
        return self._candidates

    @property
    def dataframe(self) -> DataFrame:
        return self._df

    def x(self, kind: Optional[str] = "train") -> np.ndarray:
        return np.array(self.dataframe[self._df[kind]]["Features"].values.tolist())

    def y(self, kind="train") -> np.ndarray:
        return self.dataframe[self.dataframe[kind]]["Labels"].values

    def save(self, save_path: str) -> str:
        self.dataframe.to_csv(save_path, index=False)

        return save_path

    @classmethod
    def load(cls, file_path: str, ref: Optional[DataFrame] = None, candidates: Optional[AnchoredOntoMappings] = None) -> "MlpDataset":
        return cls(dataframe=pd.read_csv(file_path), ref=ref, candidates=candidates)
        


    
