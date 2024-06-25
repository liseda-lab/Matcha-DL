
import os
from matcha_dl.core.actions.Evaluation import BioMLEvaluationAction

def main():
    data_url = "https://zenodo.org/records/8193375/files/bio-ml.zip?download=1"  
    output_dir = "output_dir"  

    BioMLEvaluationAction.run(data_url, output_dir)

if __name__ == "__main__":
    main()