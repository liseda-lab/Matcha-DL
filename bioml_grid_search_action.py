
import itertools
from pathlib import Path
from matcha_dl.core.actions.Evaluation import BioMLEvaluationAction
import os
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor

def generate_config_file(
                         base_dir,
                         experiment_dir,
                         file_name,
                         combo
                        ):

    nr_negatives = combo["nr_negatives"]
    DL_threshold = combo["DL_threshold"]
    matcha_cardinality = combo["matcha_cardinality"]
    matcha_threshold = combo["matcha_threshold"]
    nr_epochs = combo["nr_epochs"]
    batch_size = combo["batch_size"]
    model_type = "MlpClassifier"
    model_params = combo["model_params"]
    loss_type = combo["loss_type"]
    loss_params = combo["loss_params"]
    optimizer = combo["optimizer"]
    optimizer_params = combo["optimizer_params"]
    content = f"""
number_of_negatives: {nr_negatives}

use_last_checkpoint: False

threshold: {DL_threshold}

matcha_params:
  max_heap: 64G
  cardinality: {matcha_cardinality}
  threshold: {matcha_threshold}
  
training_params:
  epochs: {nr_epochs}
  batch_size: {batch_size}
  save_interval: 5
  
model:
  name: {model_type}
  params:
"""
    for param, value in model_params.items():
        content += f"    {param}: {value}\n"
    content += f"""
loss:
  name: {loss_type}
  params:
"""
    for param, value in loss_params.items():
        content += f"    {param}: {value}\n"
    content += f"""
optimizer:
  name: {optimizer}
  params:
"""
    for param, value in optimizer_params.items():
        content += f"    {param}: {value}\n"

    file_path = Path(base_dir) / Path(experiment_dir) / Path(file_name)
    with open(file_path, "w") as f:
        f.write(content)

def generate_combo_dir(base_dir, combo):
    nr_negatives = combo["nr_negatives"]
    DL_threshold = combo["DL_threshold"]
    matcha_cardinality = combo["matcha_cardinality"]
    matcha_threshold = combo["matcha_threshold"]
    nr_epochs = combo["nr_epochs"]
    batch_size = combo["batch_size"]
    model_type = "MlpClassifier"
    model_params = combo["model_params"]
    loss_type = combo["loss_type"]
    loss_params = combo["loss_params"]
    optimizer = combo["optimizer"]
    optimizer_params = combo["optimizer_params"]
    
    name = f"neg{nr_negatives}-dl_thresh{DL_threshold}-card{matcha_cardinality}-mt_thresh{matcha_threshold}-epochs{nr_epochs}-bs{batch_size}-model{model_type}-model_params{model_params}-loss{loss_type}-loss_params{loss_params}-opt{optimizer}-opt_params{optimizer_params}"
    complete = Path(base_dir) / Path(name)
    Path(complete).mkdir(parents=True, exist_ok=True)
    return name

def parse_folder_name(folder_name):
    pattern = r'neg(?P<Negative_samples>\d+)-dl_thresh(?P<Matcha_DL_threshold>[\d.]+)-card(?P<Cardinality>\d+)-mt_thresh(?P<Matcha_threshold>[\d.]+)-epochs(?P<Nr_epochs>\d+)-bs(?P<Batch_size>\d+)-model(?P<Model>\w+)-model_params(?P<Model_params>{.*?})-loss(?P<Loss>\w+)-loss_params(?P<Loss_params>{.*?})-opt(?P<Optimizer>\w+)-opt_params(?P<Optimizer_params>{.*?})'
    match = re.match(pattern, folder_name)
    if match:
        return match.groupdict()
    else:
        return {}

def parse_results_file(file_path, config_filenames):
    with open(file_path, 'r') as file:
        content = file.read()
    
    tasks = content.strip().split("\n\n")
    parsed_data = []
    
    for task in tasks:
        lines = task.strip().split("\n")
        task_name = lines[0].split(": ")[1].split(" ")[0]
        for line in lines[1:]:
            match = re.match(r'(\w+-supervised) (\w+ \w+ Results): (.+)', line)
            if match:
                supervision_type = match.group(1)
                result_type = match.group(2)
                if not "N/A" in line:
                    metrics = eval(match.group(3))
                else:
                    metrics = None
                metrics_data = {
                    'Task': task_name,
                    'Supervision': supervision_type,
                    'Result Type': result_type,
                }
                if 'Global' in result_type and "global" in config_filenames and metrics:
                    metrics_data.update({
                        'Precision': metrics.get('P', None),
                        'Recall': metrics.get('R', None),
                        'F1': metrics.get('F1', None)
                    })
                else:
                    metrics_data.update({
                        'Precision': "N/A",
                        'Recall': "N/A",
                        'F1': "N/A"
                    })
                if 'Local' in result_type and "local" in config_filenames and metrics:
                    metrics_data.update({
                        'MRR': metrics.get('MRR', None),
                        'Hits@1': metrics.get('Hits@1', None),
                        'Hits@5': metrics.get('Hits@5', None),
                        'Hits@10': metrics.get('Hits@10', None)
                    })
                else:
                    metrics_data.update({
                        'MRR': "N/A",
                        'Hits@1':"N/A",
                        'Hits@5': "N/A",
                        'Hits@10': "N/A"
                    })
                parsed_data.append(metrics_data)
    
    return parsed_data

def write_results_xls(base_dir, config_filenames):
  
    all_data = []
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            results_file_path = os.path.join(folder_path, 'results.txt')
            if os.path.exists(results_file_path):
                folder_data = parse_folder_name(folder_name)
                parsed_data = parse_results_file(results_file_path, config_filenames)
                for data in parsed_data:
                    data.update(folder_data)
                    all_data.append(data)
    
    global_columns = [
        'Negative_samples', 'Matcha_DL_threshold', 'Cardinality', 'Matcha_threshold',
        'Nr_epochs', 'Batch_size', 'Model', 'Model_params', 'Loss', 'Loss_params',
        'Optimizer', 'Optimizer_params', 'Task', 'Supervision', 'Result Type',
        'Precision', 'Recall', 'F1'
    ]
    
    local_columns = [
        'Negative_samples', 'Matcha_DL_threshold', 'Cardinality', 'Matcha_threshold',
        'Nr_epochs', 'Batch_size', 'Model', 'Model_params', 'Loss', 'Loss_params',
        'Optimizer', 'Optimizer_params', 'Task', 'Supervision', 'Result Type',
        'MRR', 'Hits@1', 'Hits@5', 'Hits@10'
    ]

    df = pd.DataFrame(all_data)
    
    # Separate the results into Local and Global
    local_results = df[df['Result Type'].str.contains('Local')]
    global_results = df[df['Result Type'].str.contains('Global')]
    
    # Write to Excel with separate sheets
    with pd.ExcelWriter('results_summary.xlsx') as writer:
        global_results.to_excel(writer, sheet_name='Global Results', index=False, columns=global_columns)
        local_results.to_excel(writer, sheet_name='Local Results', index=False, columns=local_columns)

# for combo in combinations:
  
#     # Creates a directory for that combo
#     combo_dir = generate_combo_dir(base_dir, combo)

#     # Generates a config file in that directory for that combo
#     generate_config_file(base_dir, combo_dir, CONFIG_FILENAME, combo)

#     # Pass config_filenames (used to only generate either only local or global alignments)
#     if SCOPE == "local":
#       config_filenames = {"local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
#     elif SCOPE == "global":
#       config_filenames = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
#     elif SCOPE == "both":
#       config_filenames = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}, "local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}

#     combo_dir = Path(base_dir) / Path(combo_dir)
    
#     # Runs alignment and evaluation using that configuration
#     BioMLEvaluationAction.run("https://zenodo.org/records/8193375/files/bio-ml.zip?download=1" , combo_dir, config_filenames)

def process_combo(combo):

    # Creates a directory for that combo
    combo_dir = generate_combo_dir(base_dir, combo)

    # Generates a config file in that directory for that combo
    generate_config_file(base_dir, combo_dir, CONFIG_FILENAME, combo)

    # Pass config_filenames (used to only generate either only local or global alignments)
    if SCOPE == "local":
        config_filenames = {"local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
    elif SCOPE == "global":
        config_filenames = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
    elif SCOPE == "both":
        config_filenames = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}, "local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}

    combo_dir = Path(base_dir) / Path(combo_dir)
    
    # Runs alignment and evaluation using that configuration
    BioMLEvaluationAction.run("https://zenodo.org/records/8193375/files/bio-ml.zip?download=1", combo_dir, config_filenames)
    
    # Return any necessary data, for example, config_filenames for further processing
    return config_filenames

# Assuming `combinations` is defined and accessible
if __name__ == "__main__":

    # Define the parameter grids

    PARAM_GRID_local = {
        "nr_negatives": [1],
        "DL_threshold": [0.7],
        "matcha_cardinality": [50],
        "matcha_threshold": [0.1],
        "nr_epochs": [5, 10],
        "batch_size": [1],
        "model_params": [{"layers": [128, 256]}, {"layers": [64, 128]}],
        "loss_type": ["BCELossWeighted"],
        "loss_params": [{"weight": [0.01, 0.99]}, {"weight": [0.02, 0.98]}],
        "optimizer": ["Adam", "SGD"],
        "optimizer_params": [{"lr": 0.001}, {"lr": 0.01}]
    }

    PARAM_GRID_global_inst_1 = {
        "nr_negatives": [1],
        "DL_threshold": [0.5, 0.7, 0.9],
        "matcha_cardinality": [5, 10, 30, 50],
        "matcha_threshold": [0.1, 0.3],
        "nr_epochs": [5, 10, 20],
        "batch_size": [1, 10],
        "model_params": [{"layers": [128, 256]}, {"layers": [64, 128]}],
        "loss_type": ["BCELossWeighted"],
        "loss_params": [{"weight": [0.5, 0.5]}],
        "optimizer": ["Adam", "SGD"],
        "optimizer_params": [{"lr": 0.001}, {"lr": 0.01}]
    }

    PARAM_GRID_global_inst_2 = {
        "nr_negatives": [99],
        "DL_threshold": [0.5, 0.7, 0.9],
        "matcha_cardinality": [5, 10, 30, 50],
        "matcha_threshold": [0.1, 0.3],
        "nr_epochs": [5, 10, 20],
        "batch_size": [1, 10],
        "model_params": [{"layers": [128, 256]}, {"layers": [64, 128]}],
        "loss_type": ["BCELossWeighted"],
        "loss_params": [{"weight": [0.01, 0.99]}],
        "optimizer": ["Adam", "SGD"],
        "optimizer_params": [{"lr": 0.001}, {"lr": 0.01}]
    }

    SPECIFIC_PARAM_GRID_GLOBAL = {
        "nr_negatives": [1],
        "DL_threshold": [0.7],
        "matcha_cardinality": [10, 20],
        "matcha_threshold": [0.1],
        "nr_epochs": [20],
        "batch_size": [1, 64],
        "model_params": [{"layers": [128, 256, 128]}],
        "loss_type": ["BCELossWeighted"],
        "loss_params": [{"weight": [0.5, 0.5]}],
        "optimizer": ["Adam"],
        "optimizer_params": [{"lr": 0.001}]
    }


    CONFIG_FILENAME = "config.yaml"
    SCOPE = "global"
    PARAM_GRID = PARAM_GRID_global_inst_1

    # Generate all combinations of parameters
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    base_dir = "specific_experiment_09_07"

    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Necessary to guard the entry point of the multiprocessing application to avoid recursive spawning of subprocesses on Windows
    with ProcessPoolExecutor() as executor:
        # Map each combo to the process_combo function and execute in parallel
        config_filenames = list(executor.map(process_combo, combinations, SCOPE, CONFIG_FILENAME, base_dir))
    
    # Process results, e.g., aggregate and write to an Excel file
    # This step depends on how you need to use the results returned by process_combo
    # Example: write_results_xls(base_dir, aggregate_results(results))

    write_results_xls(base_dir, config_filenames)