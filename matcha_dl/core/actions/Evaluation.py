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
import traceback
from datetime import datetime

def copy_and_rename_file(src_file_path: str, dest_dir: str, new_file_name: str) -> None:
    src_path = Path(src_file_path)
    dest_dir_path = Path(dest_dir)

    if not dest_dir_path.exists():
        dest_dir_path.mkdir(parents=True, exist_ok=True)

    dest_file_path = dest_dir_path / new_file_name
    shutil.copy(src_path, dest_file_path)
    
    
# TODO: Tem de receber dict de config files {"unsup": {"local": config_local_unsup_name, "global": config_global_unsup_name}}
# se uma das scopes não tiver config file, não corre (acrescentar N/A à tabela no fim)
class BioMLEvaluationAction(Protocol):
    @staticmethod
    def run(
        data_url: str,
        output_dir: str,
        hierarchical : dict[str : dict[str : str]],
    ) -> None:

        q = Path(output_dir)
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

        scopes = list(hierarchical.keys())
        supervision_cases = set()

        # Step 2: Iterate through the outer dictionary
        for inner_dict in hierarchical.values():
            # Step 3: Update the set with keys from each inner dictionary
            supervision_cases.update(inner_dict.keys())
        print(supervision_cases)
        semi_sup_path = submission_path / Path("semi-supervised_submission")
        unsup_path = submission_path / Path("unsupervised_submission")
        if not semi_sup_path.exists():
            os.mkdir(semi_sup_path)
        if not unsup_path.exists():
            os.mkdir(unsup_path)
        translation = {"sup": semi_sup_path, "unsup": unsup_path}
        
        for task in tasknames:
            if not task.startswith("."):
                try:
                    example_ds = BioMLDataset((root_downloaded_folder / task).resolve())

                    ignored_index = get_ignored_class_index(Ontology(example_ds._source))
                    ignored_index.update(get_ignored_class_index(Ontology(example_ds._target)))

                    print("Matching", example_ds._source, "and", example_ds._target)
                    
                    for case in supervision_cases: 
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
                            if "local" in hierarchical:
                                runner = AlignmentRunner(
                                    source_ontology_file=example_ds._source,
                                    target_ontology_file=example_ds._target,
                                    output_dir=local_dir,
                                    reference_file=example_ds._reference,
                                    candidates_file=example_ds._candidates,
                                    config_file=q / hierarchical["local"][case]
                                )
                                runner.run()
                            if "global" in hierarchical:
                                # Global
                                runner = AlignmentRunner(
                                    source_ontology_file=example_ds._source,
                                    target_ontology_file=example_ds._target,
                                    output_dir=global_dir,
                                    reference_file=example_ds._reference,
                                    config_file=q / hierarchical["global"][case]
                                )
                                runner.run()
                        elif case == "unsup":
                            # No reference file
                            # Local
                            if "local" in hierarchical:
                                runner = AlignmentRunner(
                                    source_ontology_file=example_ds._source,
                                    target_ontology_file=example_ds._target,
                                    output_dir=local_dir,
                                    candidates_file=example_ds._candidates,
                                    config_file=q / hierarchical["local"][case]
                                )
                                runner.run()
                            if "global" in hierarchical:
                                # Global
                                runner = AlignmentRunner(
                                    source_ontology_file=example_ds._source,
                                    target_ontology_file=example_ds._target,
                                    output_dir=global_dir,
                                    config_file=q / hierarchical["global"][case]
                                )
                                runner.run()
                            
                    # Do evaluation
                    results = BioMLResults(output_dir / Path("alignment_results") / task)
                    
                    if "global" in hierarchical:
                        if "unsup" in hierarchical["global"]:
                            unsupervised_global_alignment = str(results.unsup.global_scope.alignment)
                            unsupervised_match_results = matching_eval(unsupervised_global_alignment, str(example_ds._full_reference.resolve()), None, ignored_index, 0.0)
                        else:
                            unsupervised_match_results = "N/A"
                        if "sup" in hierarchical["global"]:
                            semisupervised_global_alignment = str(results.sup.global_scope.alignment)
                            semisupervised_match_results = matching_eval(semisupervised_global_alignment, str(example_ds._full_reference.resolve()), str(example_ds._reference.resolve()), ignored_index, 0.0)
                        else:
                            semisupervised_match_results = "N/A"
                    else:
                        unsupervised_match_results = "N/A"
                        semisupervised_match_results = "N/A"
                    # Local matching evaluation
                    if "local" in hierarchical:
                        if "unsup" in hierarchical["global"]:
                            unsupervised_local_alignment = str(results.unsup.local_scope.alignment)
                            unsupervised_rank_results = ranking_eval(unsupervised_local_alignment, Ks=[1, 5, 10])
                        else:
                            unsupervised_rank_results = "N/A"
                        if "sup" in hierarchical["local"]:    
                            semisupervised_local_alignment = str(results.sup.local_scope.alignment)
                            semisupervised_rank_results = ranking_eval(semisupervised_local_alignment, Ks=[1, 5, 10])
                        else:
                            semisupervised_rank_results = "N/A"
                    else:
                        unsupervised_rank_results = "N/A"
                        semisupervised_rank_results = "N/A"
                    
                    
                    # alignments  = {"sup": [semisupervised_local_alignment, semisupervised_global_alignment], "unsup": [unsupervised_local_alignment, unsupervised_global_alignment]}
                    # # Write alignment in submission folder
                    # for case in supervision_cases:
                    #   target_folder = translation[case]
                    #   local_alignment, global_alignment = alignments[case]
                    #   converted_task = task.replace("-", "2")
                    #   submission_task_folder = target_folder / Path(converted_task)
                    #   if not submission_task_folder.exists():
                    #     os.mkdir(submission_task_folder)
                    
                    #   copy_and_rename_file(global_alignment, submission_task_folder, "match.result.tsv")
                    #   copy_and_rename_file(local_alignment, submission_task_folder, "rank.result.tsv")
                
                    # Add task name and evaluation results to results.txt
                    with open(Path(output_dir) / "results.txt", "a") as results_file:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        results_file.write(f"Task: {task} {current_time}\n")
                        results_file.write(f"Unsupervised Global Matching Results: {unsupervised_match_results}\n")
                        results_file.write(f"Semi-supervised Global Matching Results: {semisupervised_match_results}\n")
                        results_file.write(f"Unsupervised Local Ranking Results: {unsupervised_rank_results}\n")
                        results_file.write(f"Semi-supervised Local Ranking Results: {semisupervised_rank_results}\n\n")

                except Exception as e:
                    error_message = f"Error processing task {task}: {str(e)}"
                    with open(Path(output_dir) / "error_log.txt", "a") as error_log_file:
                        error_log_file.write(error_message + "\n")
                        # Write the full stack trace to the log file
                        traceback.print_exc(file=error_log_file)
                    print(error_message)  # Optionally print to console for immediate feedback
        # Delete ds_download_folder
        os.rmdir(ds_download_folder)
              
