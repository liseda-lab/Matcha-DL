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

# Aux function
def copy_and_rename_file(src_file_path: str, dest_dir: str, new_file_name: str) -> None:
    src_path = Path(src_file_path)
    dest_dir_path = Path(dest_dir)

    if not dest_dir_path.exists():
        dest_dir_path.mkdir(parents=True, exist_ok=True)

    dest_file_path = dest_dir_path / new_file_name
    shutil.copy(src_path, dest_file_path)
    
class GlobalBioMLEvaluationAction(Protocol):
    @staticmethod
    def run(
        data_url: str,      # The URL of the dataset with the ontologies
        output_dir: str,
        yaml_config,
        sup_case
    ) -> None:
        q = Path(".")       
        shutil.rmtree(q / Path("ds_download"), ignore_errors=True)
        
        # Download data from URL
        #########################################################################################################################
 
        ds_download_folder = (q / Path("ds_download")).resolve()
        if not ds_download_folder.exists():
            os.mkdir(ds_download_folder)
        # 1- Download and unzip url
        zip_path, _ = urllib.request.urlretrieve(data_url)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(ds_download_folder)
        
        root_downloaded_folder = ds_download_folder / os.listdir(ds_download_folder.resolve())[0]

        # Do matching for each task
        #########################################################################################################################
        tasknames = [x for x in os.listdir(root_downloaded_folder)]
        
        if not Path(output_dir).exists():
            os.mkdir(Path(output_dir))
        submission_path = (Path(output_dir) / Path("submission"))
        if not submission_path.exists():
            os.mkdir(submission_path)

        semi_sup_path = submission_path / Path("semi-supervised_submission")
        unsup_path = submission_path / Path("unsupervised_submission")
        
        if not semi_sup_path.exists():
            os.mkdir(semi_sup_path)
        if not unsup_path.exists():
            os.mkdir(unsup_path)
        translation = {"sup": semi_sup_path, "unsup": unsup_path}
        
        shutil.rmtree(output_dir, ignore_errors=True)
        
        for task in tasknames[:1]:
            example_ds = BioMLDataset((root_downloaded_folder / task).resolve())

            ignored_index = get_ignored_class_index(Ontology(example_ds._source))
            ignored_index.update(get_ignored_class_index(Ontology(example_ds._target)))

            print("Matching", example_ds._source, "and", example_ds._target)
            
            case = sup_case
            
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
                
            global_dir = task_result_dir / Path("global_scope")
            if not global_dir.exists():
                os.mkdir(global_dir.resolve())
            local_dir = task_result_dir / Path("local_scope")
            if not local_dir.exists():
                os.mkdir(local_dir.resolve())
            

            if case == "sup":

                # Global
                runner = AlignmentRunner(
                    source_ontology_file=example_ds._source,
                    target_ontology_file=example_ds._target,
                    output_dir=global_dir,
                    reference_file=example_ds._reference,
                    config_file=yaml_config
                )
                runner.run()
            elif case == "unsup":

                # Global
                runner = AlignmentRunner(
                    source_ontology_file=example_ds._source,
                    target_ontology_file=example_ds._target,
                    output_dir=global_dir,
                    config_file=yaml_config
                )
                runner.run()
                    
            # Do evaluation for that task
            #########################################################################################################################
            results = BioMLResults(output_dir / Path("alignment_results") / task)
            
            
            if sup_case == "unsup":
            # Run global matching evaluation
                unsupervised_global_alignment = str(results.unsup.global_scope.alignment)
                unsupervised_match_results = matching_eval(unsupervised_global_alignment, str(example_ds._full_reference.resolve()), None, ignored_index, 0.0)
                results = [f"Unsupervised Global Matching Results: {unsupervised_match_results}\n\n"]
            elif sup_case == "sup":
                semisupervised_global_alignment = str(results.sup.global_scope.alignment)
                semisupervised_match_results = matching_eval(semisupervised_global_alignment, str(example_ds._full_reference.resolve()), str(example_ds._reference.resolve()), ignored_index, 0.0)
                results = [f"Semi-supervised Global Matching Results: {semisupervised_match_results}\n\n"]
            
            # Write evaluation results to txt file
            #########################################################################################################################
            with open(Path(output_dir) / "results.txt", "a") as results_file:
                results_file.write(f"Task: {task}\n")
                for r in results:
                    results_file.write(r)
                    

              
