from ast import literal_eval
from typing import Dict, List, Union

import pandas as pd

from matcha_dl.core.contracts.processor import IProcessor
from matcha_dl.core.entities.dataset import MlpDataset


class MainProcessor(IProcessor):

    def _process(self) -> pd.DataFrame:
        """Processes the data.

        Returns:
            pd.DataFrame: The processed data.
        """

        if self.refs is not None:

            # get training set
            self.log("Creating Training Set...", level="debug")

            # get positive samples from refs
            positive_set = self.refs

            # get negative samples from sampler
            self.log("#Sampling Negative Samples...", level="debug")
            negative_set = pd.DataFrame(
                self.sampler.sample(positive_set.SrcEntity, positive_set.TgtEntity),
                columns=["SrcEntity", "TgtEntity", "Score"],
            )

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

            # Inference set

            # get all sources not in refs
            self.log("Creating Inference Set...", level="debug")
            self.log(
                "#Getting sources for inference (matcha - refs) #assuming global align",
                level="debug",
            )

            inference_sources = set(self.matcha_scores.keys()) - set(
                self.refs["SrcEntity"].unique()
            )

        else:

            self.log("Creating Inference Set...", level="debug")
            self.log("#Getting all sources from matcha #assuming global align", level="debug")

            # if no refs, get all sources from matcha

            inference_sources = set(self.matcha_scores.keys())

        if self.candidates is not None:

            # if get sources from candidates instead of refs
            self.log("#Local Alignment", level="debug")

            # In the current implementation, this block is not usefull since if candidates exist
            # the inference sources are not used, but the candidates are used directly

            # self.log("##Getting sources for inference (candidates)", level="debug")

            # inference_sources = self.candidates.SrcEntity

        self.log("#Getting candidates from sources", level="debug")
        inference_set = pd.DataFrame(
            self._get_cands(inference_sources), columns=["SrcEntity", "TgtEntity", "Score"]
        )

        # get scores features from matcha

        self.log("#Getting Scores...", level="debug")

        inference_set = self._get_scores(inference_set)

        # assign inference label

        inference_set["train"] = False
        inference_set["inference"] = True

        # combine training and inference sets

        if self.refs is not None:

            self.log("#Combining Training and Inference Sets...", level="debug")

            dataset = pd.concat([training_set, inference_set], ignore_index=True)

        else:
            dataset = inference_set

        dataset.rename(columns={"Score": "Labels"}, inplace=True)

        self.log("#Processing Done", level="debug")

        return MlpDataset(dataset, ref=self.refs, candidates=self.candidates)

    def _matcha_scores_to_dict(self, csv_file: str) -> Dict:
        """Reads matcha scores file and parses it into a dict.

        Args:
            csv_file (str): The CSV file.

        Returns:
            Dict: The dictionary of matcha scores.
        """

        df = pd.read_csv(csv_file)

        return {
            src_ent: {
                row["Entity 2"]: row[["LM", "WM", "SM", "BKM", "LLMM"]].values.tolist()
                for _, row in df[df["Entity 1"] == src_ent].iterrows()
            }
            for src_ent in df["Entity 1"].unique()
        }

    def _get_scores(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Adds matcha scores to the dataset.

        Args:
            dataset (pd.DataFrame): The dataset.

        Returns:
            pd.DataFrame: The dataset with scores.
        """

        feats = []
        flag = 0

        for _, row in dataset.iterrows():
            scores = self.matcha_scores.get(row["SrcEntity"], {}).get(row["TgtEntity"], [])

            if scores:
                feats.append(scores)
                flag += 1
            else:
                feats.append(self.random.uniform(low=0.0, high=0.4, size=(5,)).tolist())

        dataset["Features"] = feats

        return dataset

    def _get_cands(self, sources: List[str]) -> List[List[Union[str, int]]]:
        """Gets candidates from matcha for global matching, or candidates from file for ranking (local matching).

        Args:
            sources (List[str]): The sources.

        Returns:
            List[List[Union[str, int]]]: The candidates.
        """

        if self.candidates is not None:

            # Local Matching Candidates
            # Retrieved from candidates input file

            return [
                [source, cand, 0]
                for source, _, target_cands in self.candidates.values
                for cand in literal_eval(target_cands)
            ]

        else:

            # Global Matching Candidates
            # Retrieved from matcha

            return [
                [source, cand, 0]
                for source in sources
                for cand in self.matcha_scores.get(source).keys()
            ]
