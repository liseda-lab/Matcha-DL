from typing import Protocol
from matcha_dl.core.entities.BioMLDataset import BioMLDataset
from matcha_dl.core.entities.BioMLSubmission import BioMLResults
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
    
def download_data(output_dir, data_url):
    q = Path(output_dir)
    ds_download_folder = (q / Path("ds_download")).resolve()
    if not ds_download_folder.exists():
        os.mkdir(ds_download_folder)
    
    # 1- Download and unzip url
    zip_path, _ = urllib.request.urlretrieve(data_url)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(ds_download_folder)
    return ds_download_folder

def read_task_names(root_downloaded_folder):
    
    return [x for x in os.listdir(root_downloaded_folder)]

def generate_ignored_index(task_dataset_folder):
    ignored_index = get_ignored_class_index(Ontology(task_dataset_folder._source))
    ignored_index.update(get_ignored_class_index(Ontology(task_dataset_folder._target)))
    return ignored_index

def generate_submission_tree(output_dir):
        submission_path = (output_dir / Path("submission"))
        
        if not submission_path.exists():
            os.mkdir(submission_path)
        
        semi_sup_path = submission_path / Path("semi-supervised_submission")
        unsup_path = submission_path / Path("unsupervised_submission")
        if not semi_sup_path.exists():
            os.mkdir(semi_sup_path)
        if not unsup_path.exists():
            os.mkdir(unsup_path)
            
        return semi_sup_path, unsup_path
    
def generate_results_tree(output_dir, task, case):
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
    return global_dir, local_dir
    
def run_alignments(output_dir, case, config_filenames, task_dataset_folder, global_dir, local_dir):
    if case == "sup":
        # Give reference file
        # Local
        if "local" in config_filenames:
            runner = AlignmentRunner(
                source_ontology_file=task_dataset_folder._source,
                target_ontology_file=task_dataset_folder._target,
                output_dir=local_dir,
                reference_file=task_dataset_folder._reference,
                candidates_file=task_dataset_folder._candidates,
                config_file=Path(output_dir) / config_filenames["local"][case]
            )
            runner.run()
        if "global" in config_filenames:
            # Global
            runner = AlignmentRunner(
                source_ontology_file=task_dataset_folder._source,
                target_ontology_file=task_dataset_folder._target,
                output_dir=global_dir,
                reference_file=task_dataset_folder._reference,
                config_file=Path(output_dir) / config_filenames["global"][case]
            )
            runner.run()
    elif case == "unsup":
        # No reference file
        # Local
        if "local" in config_filenames:
            runner = AlignmentRunner(
                source_ontology_file=task_dataset_folder._source,
                target_ontology_file=task_dataset_folder._target,
                output_dir=local_dir,
                candidates_file=task_dataset_folder._candidates,
                config_file=Path(output_dir) / config_filenames["local"][case]
            )
            runner.run()
        if "global" in config_filenames:
            # Global
            runner = AlignmentRunner(
                source_ontology_file=task_dataset_folder._source,
                target_ontology_file=task_dataset_folder._target,
                output_dir=global_dir,
                config_file=Path(output_dir) / config_filenames["global"][case]
            )
            runner.run()

def run_evaluations(config_filenames, results, task_dataset_folder, ignored_index):
    semisupervised_global_alignment = None
    unsupervised_global_alignment = None
    semisupervised_local_alignment = None
    unsupervised_local_alignment = None
    if "global" in config_filenames:
        if "unsup" in config_filenames["global"]:
            unsupervised_global_alignment = str(results.unsup.global_scope.alignment)
            unsupervised_match_results = matching_eval(unsupervised_global_alignment, str(task_dataset_folder._full_reference.resolve()), None, ignored_index, 0.0)
        else:
            unsupervised_match_results = "N/A"
        if "sup" in config_filenames["global"]:
            semisupervised_global_alignment = str(results.sup.global_scope.alignment)
            semisupervised_match_results = matching_eval(semisupervised_global_alignment, str(task_dataset_folder._full_reference.resolve()), str(task_dataset_folder._reference.resolve()), ignored_index, 0.0)
        else:
            semisupervised_match_results = "N/A"
    else:
        unsupervised_match_results = "N/A"
        semisupervised_match_results = "N/A"
    # Local matching evaluation
    if "local" in config_filenames:
        if "unsup" in config_filenames["local"]:
            unsupervised_local_alignment = str(results.unsup.local_scope.alignment)
            unsupervised_rank_results = ranking_eval(unsupervised_local_alignment, Ks=[1, 5, 10])
        else:
            unsupervised_rank_results = "N/A"
        if "sup" in config_filenames["local"]:    
            semisupervised_local_alignment = str(results.sup.local_scope.alignment)
            semisupervised_rank_results = ranking_eval(semisupervised_local_alignment, Ks=[1, 5, 10])
        else:
            semisupervised_rank_results = "N/A"
    else:
        unsupervised_rank_results = "N/A"
        semisupervised_rank_results = "N/A"
    return (unsupervised_match_results, unsupervised_rank_results, semisupervised_match_results, semisupervised_rank_results), (unsupervised_global_alignment, unsupervised_local_alignment, semisupervised_global_alignment, semisupervised_local_alignment)

def write_results(output_dir, task, unsupervised_match_results, semisupervised_match_results, unsupervised_rank_results, semisupervised_rank_results):
    with open(Path(output_dir) / "results.txt", "a") as results_file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_file.write(f"Task: {task} {current_time}\n")
        results_file.write(f"Unsupervised Global Matching Results: {unsupervised_match_results}\n")
        results_file.write(f"Semi-supervised Global Matching Results: {semisupervised_match_results}\n")
        results_file.write(f"Unsupervised Local Ranking Results: {unsupervised_rank_results}\n")
        results_file.write(f"Semi-supervised Local Ranking Results: {semisupervised_rank_results}\n\n")
    results_file.close()
    
def write_submission(supervision_cases, supervision_to_path, task, alignments):
    
    # Write alignment in submission folder
    for case in supervision_cases:
        target_folder = supervision_to_path[case]
        local_alignment, global_alignment = alignments[case]
        converted_task = task.replace("-", "2")
        submission_task_folder = target_folder / Path(converted_task)
        if not submission_task_folder.exists():
            os.mkdir(submission_task_folder)
        
        if global_alignment:
            copy_and_rename_file(global_alignment, submission_task_folder, "match.result.tsv")
        if local_alignment:
            copy_and_rename_file(local_alignment, submission_task_folder, "rank.result.tsv")
        
class BioMLEvaluationAction(Protocol):
    @staticmethod
    def run(
        data_url: str,
        output_dir: str,
        config_filenames : dict[str : dict[str : str]],
        do_submission : bool = True,
    ) -> None:

        # Download datasets
        ds_download_folder = download_data(output_dir, data_url)

        root_downloaded_folder = ds_download_folder / os.listdir(ds_download_folder.resolve())[0]
        
        # Read task names within
        tasknames = read_task_names(root_downloaded_folder)

        supervision_cases = set()

        for inner_dict in config_filenames.values():
            
            supervision_cases.update(inner_dict.keys())
            
        if do_submission:
            # Generate submission tree
            semi_sup_path, unsup_path = generate_submission_tree(output_dir)
        
        # For each task present
        for task in tasknames:
            
            # Ignore irrelevant hidden folders
            if not task.startswith("."):
                
                try:
                    task_dataset_folder = BioMLDataset((root_downloaded_folder / task).resolve())

                    # Generate ignored index
                    ignored_index = generate_ignored_index(task_dataset_folder)

                    # For each supervision case to be taken into account
                    for case in supervision_cases: 
                        
                        global_dir, local_dir = generate_results_tree(output_dir, task, case)
  
                        run_alignments(output_dir, case, config_filenames, task_dataset_folder, global_dir, local_dir)
                            
                    # Do evaluation
                    results = BioMLResults(output_dir / Path("alignment_results") / task)
                    
                    results, alignments = run_evaluations(config_filenames, results, task_dataset_folder, ignored_index)
                    
                    unsupervised_global_alignment, unsupervised_local_alignment, semisupervised_global_alignment, semisupervised_local_alignment = alignments
                    
                    unsupervised_match_results, unsupervised_rank_results, semisupervised_match_results, semisupervised_rank_results = results
                    
                    if do_submission:
                        # Write to submission tree
                        supervision_to_path = {"sup": semi_sup_path, "unsup": unsup_path}
                        
                        alignments_organized  = {"sup": [semisupervised_local_alignment, semisupervised_global_alignment], "unsup": [unsupervised_local_alignment, unsupervised_global_alignment]}
                        
                        write_submission(supervision_cases, supervision_to_path, task, alignments_organized)
                
                    # Add task name and evaluation results to results.txt
                    write_results(output_dir, task, unsupervised_match_results, semisupervised_match_results, unsupervised_rank_results, semisupervised_rank_results)

                except Exception as e:
                    error_message = f"Error processing task {task}: {str(e)}"
                    with open(Path(output_dir) / "error_log.txt", "a") as error_log_file:
                        error_log_file.write(error_message + "\n")
                        # Write the full stack trace to the log file
                        traceback.print_exc(file=error_log_file)
                    print(error_message)  # Optionally print to console for immediate feedback
                    
        # Delete ds_download_folder
        shutil.rmtree(ds_download_folder, ignore_errors=True)
              
