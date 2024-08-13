from ast import literal_eval
from typing import Optional

import numpy as np
import pandas as pd

from pathlib import Path

DataFrame = pd.DataFrame

from .base import Dataset

from typing import Tuple


class TabularDataset(Dataset):

    def __init__(self, output_path: Path) -> None:
        super().__init__(output_path)

        self._df = None
        self._df_save_path = self.output_path / "dataset.csv"

    @property
    def dataframe(self) -> DataFrame:
        return self._df
    
    def __len__(self) -> int:
        if self.dataframe is None:
            return 0
        return len(self.dataframe)
    
    def __getitem__(self, idx: int, kind: str = "train") -> Tuple[np.ndarray, np.ndarray]:

        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        
        elif idx < 0:
            raise IndexError("Index out of bounds.")
        
        elif len(self) == 0:
            raise IndexError("Empty dataset.")
        
        return self.x(kind)[idx], self.y(kind)[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        return self.dataframe.__str__()

    def x(self, kind: Optional[str] = "train") -> np.ndarray:
        return np.array(self.dataframe[self.dataframe[kind]]["Features"].values.tolist())

    def y(self, kind="train") -> np.ndarray:
        return self.dataframe[self.dataframe[kind]]["Labels"].values

    def save(self) -> Path:

        self.dataframe.to_csv(str(self._df_save_path), index=False)

        self.log("#Saved Dataset...", level="debug")

        return self._df_save_path
    
    def load(self):
        self._df = pd.read_csv(self._df_save_path, converters={"Features": literal_eval})

        self.log("#Loaded Cached Dataset...", level="debug")

    def has_cache(self) -> bool:
        if self._cache_ok:
            return self._df_save_path.exists()
        return False

    def process(self) -> None:

        if self.has_cache():
            self.load()
            return
        
        inference_set = pd.DataFrame(
            self._get_cands(inference_sources), columns=["SrcEntity", "TgtEntity", "Score"]
        )
        
        if self.reference is not None:

            # get training set
            self.log("Creating Training Set...", level="debug")

            # get positive samples from refs
            positive_set = self.reference

            # add negatives
            self.log("#Adding Negative Samples...", level="debug")
            negative_set = self.negatives

            # combine positive and negative samples
            training_set = pd.concat([positive_set, negative_set], ignore_index=True)

            # get scores features from matcha
            self.log("#Getting Scores...", level="debug")
            training_set = self._get_scores(training_set)

            # assign training label
            training_set["train"] = True
            training_set["inference"] = False

            self.log("#Shuffling Training Set...", level="debug")

            training_set = training_set.sample(frac=1).reset_index(drop=True)

        


    def _get_cands(self) -> pd.DataFrame:

        return pd.DataFrame([
                [source, cand[0], 0, cand[1:]]
                for source, _, target_cands in self.candidates.values
                for cand in literal_eval(target_cands)
            ], columns=["Src", "Tgt", "Score", "Features"])

            

    def _get_matcha_features(self, dataset: pd.DataFrame) -> pd.DataFrame:

        feats = []
        
        for _, row in dataset.iterrows():

            try:
                vector = self.matcha_features.get(row["Src"]).get(row["Tgt"])
            except AttributeError:
                raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
            
            feats.append(vector)
            

        dataset["Features"] = feats

        return dataset

            
            


