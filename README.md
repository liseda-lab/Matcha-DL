# Matcha DL

## Description
Matcha-DL is an extension of the matching system Matcha to tackle semi-supervised tasks using machine-learning algorithms. Matcha builds upon the algorithms of the established system AgreementMakerLight with a novel broader core architecture designed to tackle long-standing challenges such as complex and holistic ontology matching. Matcha-DL uses a linear neural network that learns to rank candidate mappings proposed by Matcha by using a partial reference alignment as a training set, and using the confidence scores produced by Matcha's matching algorithms as features.

## Installation

**Still under development** :
The package is stil undergoing testing, we will have a distribution available on Pypi soon

To install Matcha DL, you can use pip:
```bash
pip install matcha-dl
```

## USAGE

### CLI

Matcha DL provides a command line interface for computing the alignment between two ontologies. Here's how you can use it:

```bash
matchadl --source_ontology_file <source_file_path> --target_ontology_file <target_file_path> --output_dir <output_dir_path> [--reference_file <reference_file_path>] [--candidates_file <candidates_file_path>] [--config_file <config_file_path>]
```

### API

#### AlignmentRunner Class

The `AlignmentRunner` is a quick and easy class that can be programaticly imported to run an alignemnt using Matcha-DL.

Here's an example on how to use it:

```python

from matcha_dl import AlignmentRunner

runner = AlignmentRunner(
    source_ontology_file="path/to/source_ontology_file",
    target_ontology_file="path/to/target_ontology_file",
    output_dir="path/to/output_dir",
    reference_file="path/to/reference_file",
    candidates_file="path/to/candidates_file",
    config_file="path/to/config_file"
)

runner.run()

```

### Arguments

* --source_ontology_file or -s: Path to the source ontology file (required)
* --target_ontology_file or -t: Path to the target ontology file (required)
* --output_dir or -o: Path to the output directory (required)
* --reference_file or -r: Path to the reference file (optional)
* --candidates_file or -c: Path to the candidates file (optional)
* --config_file or -C: Path to the config file (optional)

#### Details
 
* The reference file should be a reference alignment, that follows the standards from the [OAEI's Bio-ML track](https://krr-oxford.github.io/DeepOnto/bio-ml/#oaei-bio-ml-2023).
* The candidates file should be a list of all the candidates for each of the entities to rank, that follows the standards from the [OAEI's Bio-ML track](https://krr-oxford.github.io/DeepOnto/bio-ml/#oaei-bio-ml-2023).
* The configuration file is a user-defined configuration file. An example can be found at [https://github.com/liseda-lab/Matcha-DL/blob/main/matcha_dl/default_config.yaml](https://github.com/liseda-lab/Matcha-DL/blob/main/matcha_dl/default_config.yaml).

### Tasks

#### Supervised/Unsupervised settings

If reference files are provided Matcha-DL will train a model to predict an alignment, otherwise it will use the scores from matcha to compute the alignment directly.

#### Global Alignment/ Local Alignment

If a candidates file is provided Matcha-DL will generate a ranking for those candidates (local alignment), otherwise it will perform global pairwise alignemnt for all the entities in the source and target ontologies.

## Acknowledgements

This work was supported by FCT through the fellowships 2022.10557.BD (Pedro Cotovio) and 2022.11895.BD (Marta Silva), and through the LASIGE Research Unit, ref. UIDB/00408/2020 (https://doi.org/10.54499/UIDB/00408/2020) and ref. UIDP/00408/2020 (https://doi.org/10.54499/UIDP/00408/2020). It was also partially supported by the KATY project which has received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No 101017453. This work was also supported partially by project 41, HfPT: Health from Portugal, funded by the Portuguese Plano de Recuperação e Resiliência.

## Contributions

**Authors:**

Pedro Giesteira Cotovio, [1], Lucas Ferraz, [1], Daniel Faria, [2], Laura Balbi, [1], Marta Contreiras Silva, [1], Catia Pesquita, [1]

**Institutions:**

1. LASIGE, Faculdade de Ciências, Universidade de Lisboa, Portugal
2. INESC-ID, Instituto Superior Técnico, Universidade de Lisboa, Portugal




