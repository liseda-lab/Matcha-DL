
import itertools
from pathlib import Path
from matcha_dl.core.actions.Evaluation import BioMLEvaluationAction

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
  
# Define the parameter grid
param_grid_local = {
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

param_grid_global = {
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

CONFIG_FILENAME = "config.yaml"
scope = "both"


param_grid = param_grid_local
# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Generate config files for all combinations
base_dir = "test_generate_config"
Path(base_dir).mkdir(parents=True, exist_ok=True)

for combo in combinations[:3]:
  
    # Creates a directory for that combo
    combo_dir = generate_combo_dir(base_dir, combo)

    # Generates a config file in that directory for that combo
    generate_config_file(base_dir, combo_dir, CONFIG_FILENAME, combo)

    # Pass hierarchical (serves to just evaluate cases within it)
    if scope == "local":
      hierarchical = {"local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
    elif scope == "global":
      hierarchical = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
    elif scope == "both":
      hierarchical = {"global": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}, "local": {"sup": CONFIG_FILENAME, "unsup": CONFIG_FILENAME}}
    combo_dir = Path(base_dir) / Path(combo_dir)
    
    # Runs alignment and evaluation using that configuration
    BioMLEvaluationAction.run("https://zenodo.org/records/8193375/files/bio-ml.zip?download=1" , combo_dir, hierarchical)
    
    # Saves results to results.txt inside that directory
    
    
# Looks at every experiment and reads its results.txt file


# For each result

      # Writes a line with config + task + scope + sup info and respective evaulation stats