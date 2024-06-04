
from typing import Protocol

class BioMLEvaluationAction(Protocol):
  @staticmethod
  def run(
    data_url: str,
    output_dir: str,
  ) -> None:

    1- Download and unzip url
    2- Get all subdirectories inside url dir (task names)
    3- Create instance of BioMLDataset for each                                
    4- For each task Call MatchaDL runner from API, and save the results to the correct place (BioMLresults) inside the submission output_dir (if candidates are passed it runs local otherwise global)
    5 - run oaei evaluation on dir

    (step 4) results = BioMLResults(...)

    ....

    
    example_ds = BioMLDataset(...)
    ignored_index = get_ignored_class_index(example_ds.source)
    ignored_index.update(get_ignored_class_index(example_ds.target))

    
    # run global matching evaluation
    unsupervised_match_results = matching_eval(results.unsup.example_task.global, example_ds.full_reference, None, ignored_index, 0.0)
    semisupervised_match_results = matching_eval(results.sup.example_task.global, example_ds.full_reference, example_ds.reference, ignored_index, 0.0)

    # local matching evaluation

    unsupervised_rank_results = ranking_eval(results.unsup.example_task.local, has_score=True, Ks=[1, 5, 10])
    semisupervised_rank_results = ranking_eval(results.sup.example_task.local, has_score=True, Ks=[1, 5, 10])

    6 - agregate results into table and save
        
