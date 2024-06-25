
import os
from matcha_dl.core.actions.Evaluation import BioMLEvaluationAction
from matcha_dl.core.actions.GlobalEvaluation import GlobalBioMLEvaluationAction

def main():
    
    
    
    
    data_url = "https://zenodo.org/records/8193375/files/bio-ml.zip?download=1"  
     
    for sup_case in ["sup", "unsup"]:
        for thresh in ["matcha_dl/config_thresh_05.yaml", "matcha_dl/config_thresh_09.yaml"]:
            output_dir = f"./{sup_case}_{thresh.split("_")[-1].split(".")[0]}"    
            GlobalBioMLEvaluationAction.run(data_url, output_dir, thresh, sup_case)

if __name__ == "__main__":
    main()