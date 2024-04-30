# Matcha DL

## Description
Matcha-DL is an extension of the matching system Matcha to tackle semi-supervised tasks using machine-learning algorithms. Matcha builds upon the algorithms of the established system AgreementMakerLight with a novel broader core architecture designed to tackle long-standing challenges such as complex and holistic ontology matching. Matcha-DL uses a linear neural network that learns to rank candidate mappings proposed by Matcha by using a partial reference alignment as a training set, and using the confidence scores produced by Matcha's matching algorithms as features.

## Installation
To install Matcha DL, you can use pip:
```bash
pip install matcha_dl
```

## USAGE

Matcha DL provides a command line interface for computing the alignment between two ontologies. Here's how you can use it:

```bash
matcha_dl --source_ontology_file <source_file_path> --target_ontology_file <target_file_path> --output_dir <output_dir_path> [--reference_file <reference_file_path>] [--candidates_file <candidates_file_path>] [--config_file <config_file_path>]
```

## Arguments

* --source_ontology_file or -s: Path to the source ontology file (required)
* --target_ontology_file or -t: Path to the target ontology file (required)
* --output_dir or -o: Path to the output directory (required)
* --reference_file or -r: Path to the reference file (optional)
* --candidates_file or -c: Path to the candidates file (optional)
* --config_file or -C: Path to the config file (optional)

