
from typing import Protocol
from matcha_dl.core.entities import BioMLDataset
import dload
from matcha_dl import AlignmentRunner

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

    # 3- For each task call MatchaDL runner from API, do evaluation and save the results to the 
    # correct place (BioMLresults) inside the submission output_dir (if candidates are passed it runs local otherwise global)
    tasknames = [x for x in os.walk((q / "bio-ml").resolve())]
    for task in tasknames:
      example_ds = BioMLDataset(q / task)
      ignored_index = get_ignored_class_index(example_ds.source)
      ignored_index.update(get_ignored_class_index(example_ds.target))

      # Do matching
      runner = AlignmentRunner(
          source_ontology_file=example_ds.source,
          target_ontology_file=example_ds.target,
          output_dir=output_dir,
          reference_file=example_ds.reference,
          candidates_file=example_ds.candidates,
          config_file="path/to/config_file"
      )

      runner.run()
      
      # Do evaluation
      results = BioMLResults(output_dir)
      
      # Run global matching evaluation
      unsupervised_match_results = matching_eval(results.unsup.example_task.global, example_ds.full_reference, None, ignored_index, 0.0)
      semisupervised_match_results = matching_eval(results.sup.example_task.global, example_ds.full_reference, example_ds.reference, ignored_index, 0.0)
  
      # Local matching evaluation
      matcha.match(ont1, ont2)
      unsupervised_rank_results = ranking_eval(results.unsup.example_task.local, has_score=True, Ks=[1, 5, 10])
      semisupervised_rank_results = ranking_eval(results.sup.example_task.local, has_score=True, Ks=[1, 5, 10])
  
      6 - Aggregate results into table and save
    
     
