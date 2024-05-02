from abc import abstractmethod
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from matcha_dl.core.contracts.negative_sampler import INegativeSampler
from matcha_dl.core.entities.dataset import MlpDataset
from matcha_dl.core.entities.dp.anchor_mappings import AnchoredOntoMappings

PROCESSOR = "processor"

DataFrame = pd.DataFrame


class IProcessor:
    """Abstract base class for a processor that parses all data inputs and returns a dataset as a pandas dataframe.

    Attributes:
        matcha_scores (Dict): The matcha scores.
        refs (DataFrame): The reference data.
        sampler (INegativeSampler): The sampler.
        candidates (AnchoredOntoMappings): The ranking candidates.
        random (np.random.RandomState): The random state.
    """

    def __init__(self, sampler: Optional[INegativeSampler] = None, seed: Optional[int] = 42, **kwargs):
        """

        Args:
            sampler (INegativeSampler, optional): The sampler. Defaults to None.
            seed (int, optional): The seed for the random state. Defaults to 42.
        """

        self._matcha_scores = None
        self._refs = None
        self._sampler = sampler
        self._cands = None
        self._seed = seed
        self._output_file = None

        self._logger = kwargs.get("logger")
        self._cache_ok = kwargs.get("cache_ok", True)

    @property
    def matcha_scores(self) -> Dict:
        """Gets the matcha scores.

        Returns:
            Dict: The matcha scores.
        """
        return self._matcha_scores

    @property
    def refs(self) -> DataFrame:
        """Gets the reference data.

        Returns:
            DataFrame: The reference data.
        """
        return self._refs

    @property
    def sampler(self) -> INegativeSampler:
        """Gets the sampler.

        Returns:
            INegativeSampler: The sampler.
        """
        return self._sampler

    @property
    def candidates(self) -> AnchoredOntoMappings:
        """Gets the ranking candidates.

        Returns:
            AnchoredOntoMappings: The ranking candidates.
        """
        return self._cands

    @property
    def random(self) -> np.random.RandomState:
        """Gets the random state.

        Returns:
            np.random.RandomState: The random state.
        """
        return np.random.RandomState(self._seed)

    @property
    def output_file(self) -> str:
        """Gets the output file.

        Returns:
            str: The output file.
        """
        return self._output_file
    
    @property
    def has_cache(self) -> bool:
        """
        Check if the output file exists.

        Returns:
            bool: True if the output file exists, False otherwise.
        """
        if self._cache_ok and self.output_file:
            return self.output_file.is_file()
        return False

    def process(
        self, scores_file: str, ref_file: Optional[str] = None, cands_file: Optional[str] = None, output_file: Optional[str] = None
    ) -> MlpDataset:
        """Processes the data.

        Args:
            scores_file (str): The scores file.
            ref_file (str, optional): The reference file. Defaults to None.
            cands_file (str, optional): The candidates file. Defaults to None.
            output_file (str, optional): The output file. Defaults to None.

        Returns:
            MlpDataset: The processed data.
        """

        self._output_file = Path(output_file) if output_file else None

        if ref_file:
            self._refs = pd.read_csv(str(ref_file), sep="\t")

            # if refs exist sampler must not be None

            if not self.sampler:
                raise ValueError("If ref file is provided, sampler must be provided")

        if cands_file:
            self._cands = AnchoredOntoMappings.read_table_mappings(
                str(cands_file)
            ).unscored_cand_maps()

        if self.has_cache:
            self.log(f"Cache found. Loading cached dataset from {self.output_file}")
            return MlpDataset.load(self.output_file, ref=self.refs, candidates=self.candidates)

        else:
            self.log("Processing dataset", level="debug")

            # Load scores
            self._matcha_scores = self._matcha_scores_to_dict(scores_file)

            dataset = self._process()

            if output_file:
                self.log(f"Saving dataset to {output_file}", level="debug")
                dataset.save(self.output_file)

            return dataset

    @abstractmethod
    def _process(self):
        pass

    @abstractmethod
    def _matcha_scores_to_dict(self, csv_file: str) -> Dict:
        """Reads matcha scores file and parses it into a dict.

        Args:
            csv_file (str): The CSV file.

        Returns:
            Dict: The dictionary of matcha scores.
        """
        pass

    def log(self, msg: str, level: Optional[str] = "info"):
        if self._logger:
            getattr(self._logger, level)(msg)

        else:
            print(msg)
