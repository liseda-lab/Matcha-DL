
from typing import Protocol
from matcha_dl.core.entities import BioMLDataset
import dload

class BioMLEvaluationAction(Protocol):
  @staticmethod
  def run(
    data_url: str,
    data_name : str,
    output_dir: str,
  ) -> None:
    
    q = Path(data_name)
    if not q.exists():
      q.mkdir(parents=True, exist_ok=True)
      
    # 1- Download and unzip url
    dload.save_unzip(data_url, q.resolve())

    # 3- For each task call MatchaDL runner from API, and save the results to the correct place (BioMLresults) inside the submission output_dir (if candidates are passed it runs local otherwise global)
    
    tasknames = [x for x in os.walk((q / "bio-ml").resolve())]
    for task in tasknames:
      example_ds = BioMLDataset(q / task)
      ignored_index = get_ignored_class_index(example_ds.source)
      ignored_index.update(get_ignored_class_index(example_ds.target))

      # Do matching
      
      # Run global matching evaluation
      unsupervised_match_results = matching_eval(results.unsup.example_task.global, example_ds.full_reference, None, ignored_index, 0.0)
      semisupervised_match_results = matching_eval(results.sup.example_task.global, example_ds.full_reference, example_ds.reference, ignored_index, 0.0)
  
      # Local matching evaluation
  
      unsupervised_rank_results = ranking_eval(results.unsup.example_task.local, has_score=True, Ks=[1, 5, 10])
      semisupervised_rank_results = ranking_eval(results.sup.example_task.local, has_score=True, Ks=[1, 5, 10])
  
      6 - Aggregate results into table and save
    
        
    # 4- Run oaei evaluation on dir

    (step 3) results = BioMLResults(...)

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
        
