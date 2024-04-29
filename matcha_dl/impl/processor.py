
import pandas as pd

from typing import Dict, List, Union

from matcha_dl.core.entities.dataset import MlpDataset

class MainProcessor:


    def _process(self) -> pd.DataFrame:
        """Processes the data.

        Returns:
            pd.DataFrame: The processed data.
        """
        
        if self.refs:

            # get training set

            # get positive samples from refs
            positive_set = self._refs

            # get negative samples from sampler
            negative_set = self.sampler.sample(positive_set['SrcEntity'], positive_set['TgtEntity'])

            # combine positive and negative samples
            training_set = pd.concat([positive_set, negative_set], ignore_index=True)

            # get scores features from matcha
            training_set = self._get_scores(training_set)
            
            # assign training label
            training_set['train'] = True
            training_set['inference'] = False

            training_set = training_set.sample(frac=1).reset_index(drop=True)

        # Inference set

            # get all sources not in refs

            inference_sources = set(self.matcha_scores.keys()) - set(self.refs['SrcEntity'].unique())

        else:

            # if no refs, get all sources from matcha

            inference_sources = self.matcha_scores.keys()


        if self.candidates:
            
            # if get sources from candidates instead of refs

            inference_sources = self.candidates.map_dict.keys()

        inference_set = pd.DataFrame(self._get_cands(inference_sources), columns=['SrcEntity', 'TgtEntity', 'Score'])

        # get scores features from matcha

        inference_set = self._get_scores(inference_set)

        # assign inference label

        inference_set['train'] = False
        inference_set['inference'] = True

        # combine training and inference sets

        dataset = pd.concat([training_set, inference_set], ignore_index=True)

        dataset.rename(columns={'Score': 'Labels'}, inplace=True)

        return MlpDataset(dataset, ref=self.refs, candidates=self.candidates)

    def _matcha_scores_to_dict(self, csv_file: str) -> Dict:
        """Reads matcha scores file and parses it into a dict.

        Args:
            csv_file (str): The CSV file.

        Returns:
            Dict: The dictionary of matcha scores.
        """
        
        df = pd.read_csv(csv_file)

        return {src_ent:
                    {row['Entity 2']: row[['LM', 'WM', 'SM', 'BKM', 'LLMM']].values.tolist()
                     for _, row in df[df['Entity 1'] == src_ent].iterrows()}
                for src_ent in df['Entity 1'].unique()}

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
            scores = self.matcha_scores.get(row['SrcEntity'], {}).get(row['TgtEntity'], [])

            if scores:
                feats.append(scores)
                flag += 1
            else:
                feats.append(self.random.uniform(low=0.0, high=0.4, size=(5,)).tolist())

        dataset['Features'] = feats

        return dataset

    def _get_cands(self, sources: List[str]) -> List[List[Union[str, int]]]:
        """Gets candidates from matcha for global matching, or candidates from file for ranking (local matching).

        Args:
            sources (List[str]): The sources.

        Returns:
            List[List[Union[str, int]]]: The candidates.
        """

        if self.candidates:

            # Local Matching Candidates
            # Retrieved from candidates input file

            return [[source, cand, 0] for source in sources for cand in self.candidates.map_dict.get(source)]
        
        else:
        
            # Global Matching Candidates
            # Retrieved from matcha

            return [[source, cand, 0] for source in sources for cand in self.matcha_scores.get(source).keys()]