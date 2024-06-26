from typing import Protocol
from matcha_dl.core.entities.BioMLDataset import BioMLDataset
from matcha_dl.core.entities.BioMLSubmission import BioMLResults
import dload
from matcha_dl import AlignmentRunner
from deeponto.align.oaei import *
from pathlib import Path
import urllib
import zipfile
import os
import shutil
def copy_and_rename_file(src_file_path: str, dest_dir: str, new_file_name: str) -> None:
    src_path = Path(src_file_path)
    dest_dir_path = Path(dest_dir)

    if not dest_dir_path.exists():
        dest_dir_path.mkdir(parents=True, exist_ok=True)

    dest_file_path = dest_dir_path / new_file_name
    shutil.copy(src_path, dest_file_path)
    
class BioMLEvaluationAction(Protocol):
    @staticmethod
    def run(
        data_url: str,
        data_name: str,
        output_dir: str,
        supervision,
        scope, 
        task
    ) -> None:
        
        q = Path(data_name)
        ds_download_folder = (q / Path("ds_download")).resolve()
        if not ds_download_folder.exists():
            os.mkdir(ds_download_folder)
            
        # 1- Download and unzip url
        zip_path, _ = urllib.request.urlretrieve(data_url)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(ds_download_folder)

        root_downloaded_folder = ds_download_folder / os.listdir(ds_download_folder.resolve())[0]
        
        # 3- For each task call MatchaDL runner from API, do evaluation and save the results to the 
        # correct place (BioMLresults) inside the submission output_dir (if candidates are passed it runs local otherwise global)
        tasknames = [x for x in os.listdir(root_downloaded_folder)]
        submission_path = (output_dir / Path("submission"))
        if not submission_path.exists():
            os.mkdir(submission_path)

        semi_sup_path = submission_path / Path("semi-supervised_submission")
        unsup_path = submission_path / Path("unsupervised_submission")
        if not semi_sup_path.exists():
            os.mkdir(semi_sup_path)
        if not unsup_path.exists():
            os.mkdir(unsup_path)
        translation = {"sup": semi_sup_path, "unsup": unsup_path}
        for task in tasknames:
            example_ds = BioMLDataset((root_downloaded_folder / task).resolve())

            ignored_index = get_ignored_class_index(Ontology(example_ds._source))
            ignored_index.update(get_ignored_class_index(Ontology(example_ds._target)))

            print("Matching", example_ds._source, "and", example_ds._target)
            
            for case in ["sup", "unsup"]: 
                if not Path(output_dir).exists():
                    os.mkdir(Path(output_dir).resolve())
                task_result_dir = Path(output_dir) / Path("alignment_results")
                if not task_result_dir.exists():
                    os.mkdir(task_result_dir.resolve())
                task_result_dir = task_result_dir / task
                if not task_result_dir.exists():
                    os.mkdir(task_result_dir.resolve())
                task_result_dir = task_result_dir / Path(case)

                if not task_result_dir.exists():
                    os.mkdir(task_result_dir.resolve())
                    
                global_dir = task_result_dir / Path("global")
                if not global_dir.exists():
                    os.mkdir(global_dir.resolve())
                local_dir = task_result_dir / Path("local")
                if not local_dir.exists():
                    os.mkdir(local_dir.resolve())
                
                if case == "sup":
                  # Give reference file
                  # Local
                  runner = AlignmentRunner(
                      source_ontology_file=example_ds._source,
                      target_ontology_file=example_ds._target,
                      output_dir=local_dir,
                      reference_file=example_ds._reference,
                      candidates_file=example_ds._candidates,
                      config_file="matcha_dl/default_config.yaml"
                  )
                  runner.run()
                  # Global
                  runner = AlignmentRunner(
                      source_ontology_file=example_ds._source,
                      target_ontology_file=example_ds._target,
                      output_dir=global_dir,
                      reference_file=example_ds._reference,
                      config_file="matcha_dl/default_config.yaml"
                  )
                  runner.run()
                elif case == "unsup":
                  # No reference file
                  # Local
                  runner = AlignmentRunner(
                      source_ontology_file=example_ds._source,
                      target_ontology_file=example_ds._target,
                      output_dir=local_dir,
                      candidates_file=example_ds._candidates,
                      config_file="matcha_dl/default_config.yaml"
                  )
                  runner.run()
                  # Global
                  runner = AlignmentRunner(
                      source_ontology_file=example_ds._source,
                      target_ontology_file=example_ds._target,
                      output_dir=global_dir,
                      config_file="matcha_dl/default_config.yaml"
                  )
                  runner.run()
                    
            # Do evaluation
            results = BioMLResults(output_dir / Path("alignment_results") / task)
            
            if scope == "global":
                # Run global matching evaluation
                unsupervised_global_alignment = str(getattr(getattr(results.unsup,"global").alignment, "src2tgt.maps_global.tsv").resolve())
                semisupervised_global_alignment = str(getattr(getattr(results.sup, "global").alignment, "src2tgt.maps_global.tsv").resolve())
                unsupervised_match_results = matching_eval(unsupervised_global_alignment, str(example_ds._full_reference.resolve()), None, ignored_index, 0.0)
                semisupervised_match_results = matching_eval(semisupervised_global_alignment, str(example_ds._full_reference.resolve()), str(example_ds._reference.resolve()), ignored_index, 0.0)
            elif scope == "local":
            # Local matching evaluation
                unsupervised_local_alignment = str(getattr(getattr(results.unsup,"local").alignment, "src2tgt.maps_local.tsv").resolve())
                semisupervised_local_alignment = str(getattr(getattr(results.sup, "local").alignment, "src2tgt.maps_local.tsv").resolve())
                unsupervised_rank_results = ranking_eval(unsupervised_local_alignment, Ks=[1, 5, 10])
                semisupervised_rank_results = ranking_eval(semisupervised_local_alignment, Ks=[1, 5, 10])
                
            alignments  = {"sup": [semisupervised_local_alignment, semisupervised_global_alignment], "unsup": [unsupervised_local_alignment, unsupervised_global_alignment]}
            # Write alignment in submission folder
            for case in ["sup", "unsup"]:
              target_folder = translation[case]
              local_alignment, global_alignment = alignments[case]
              converted_task = task.replace("-", "2")
              submission_task_folder = target_folder / Path(converted_task)
              if not submission_task_folder.exists():
                os.mkdir(submission_task_folder)
              
              copy_and_rename_file(global_alignment, submission_task_folder, "match.result.tsv")
              copy_and_rename_file(local_alignment, submission_task_folder, "rank.result.tsv")
        
            # Add task name and evaluation results to results.txt
            with open(Path(output_dir) / "results.txt", "a") as results_file:
                results_file.write(f"Task: {task}\n")
                results_file.write(f"Unsupervised Global Matching Results: {unsupervised_match_results}\n")
                results_file.write(f"Semi-supervised Global Matching Results: {semisupervised_match_results}\n")
                results_file.write(f"Unsupervised Local Ranking Results: {unsupervised_rank_results}\n")
                results_file.write(f"Semi-supervised Local Ranking Results: {semisupervised_rank_results}\n\n")
              

              
