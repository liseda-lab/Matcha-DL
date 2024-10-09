from abc import ABC, abstractmethod
from pathlib import Path

from typing import Optional

# from mowl.owlapi import OWLAPIAdapter

from matcha_dl.impl.dp.utils import read_table

# from jpype import java

import pandas as pd
import dgl.backend as F
import numpy as np
from typing import List, Tuple, Union, Dict

DataFrame = pd.DataFrame
DataArray = Union[np.ndarray, Tuple, List, F.tensor]

# OWLAPI imports
# from org.semanticweb.owlapi.model import OWLOntology



class Dataset(ABC):

    def __init__(self, output_path: Path, matchers: List[str], **kwargs) -> None:


        self._output_path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._matchers = matchers

        self._source = None
        self._target = None
        self._candidates = None
        self._reference = None
        self._negatives = None
        self._matcha_features = None

        self._logger = kwargs.get("logger")
        self._cache_ok = kwargs.get("cache_ok", True)

    @property
    def matchers(self) -> List[str]:
        return self._matchers

    # @property
    # def source(self) -> OWLOntology:
    #     return self._source
    
    # @property
    # def target(self) -> OWLOntology:
    #     return self._target
    
    @property
    def candidates(self) -> DataFrame:
        return self._candidates
    
    @property
    def reference(self) -> DataFrame:
        return self._reference
    
    @property
    def negatives(self) -> DataFrame:
        return self._negatives
    
    @property
    def matcha_features(self) -> Dict[str, Dict[str, List[float]]]:
        return self._matcha_features
    
    @property
    def output_path(self) -> Path:
        return self._output_path
    
    # def load_ontologies(self, source_path: Path, target_path: Path) -> None:
        
    #     adapter = OWLAPIAdapter() 
    #     owl_manager = adapter.owl_manager
    #     self._source = owl_manager.loadOntologyFromOntologyDocument(
    #         java.io.File(source_path))
        
    #     self.log("#Loaded Source...", level="debug")
        
    #     self._target = owl_manager.loadOntologyFromOntologyDocument(
    #         java.io.File(target_path))
        
    #     self.log("#Loaded Target...", level="debug")

    def load_candidates(self, file_path: Path) -> None:
        self._candidates = read_table(str(file_path))
        self._candidates.columns = ["Src", "Tgt", "Candidates"]

        self.log("#Loaded Candidates...", level="debug")

    def load_reference(self, file_path: Path) -> None:
        self._reference = read_table(str(file_path))
        self._reference.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Reference...", level="debug")

    def load_negatives(self, file_path: str) -> None:
        self._negatives = read_table(str(file_path))
        self._negatives.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Negatives...", level="debug")

    
    def load_data(self, matcha_features_file: Path) -> None:

        df = read_table(matcha_features_file)
        df.columns = ["Src", "Tgt"] + self.matchers

        self._matcha_features = {
            src_ent: {
                row["Tgt"]: row[self.matchers].values.tolist()
                for _, row in df[df["Src"] == src_ent].iterrows()
            }
            for src_ent in df["Src"].unique()
        }

        self.log("#Loaded Matcha Features...", level="debug")

    def log(self, msg: str, level: Optional[str] = "info"):
        if self._logger is not None:
            getattr(self._logger, level)(msg)

        else:
            print(msg)

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[DataArray, DataArray]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def save(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def has_cache(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def x(self, kind: Optional[str] = "train") -> DataArray:
        pass

    @abstractmethod
    def y(self, kind="train") -> DataArray:
        pass


