import subprocess
from pathlib import Path

TEST_DATA_DIR = Path("~/Matcha-DL/tests/test_data/bio-ml/ncit-doid")


def main():
    source_ontology_file = str(TEST_DATA_DIR / "ncit.owl")
    target_ontology_file = str(TEST_DATA_DIR / "doid.owl")
    output_dir = str(TEST_DATA_DIR / "test_output_supervised")
    config_file = str(TEST_DATA_DIR / "config.yaml")
    reference_file = str(TEST_DATA_DIR / "refs_equiv/train.tsv")
    candidates_file = str(TEST_DATA_DIR / "refs_equiv/test.cands.tsv")

    result = subprocess.run(
        [
            "poetry",
            "run",
            "matchadl",
            "-s",
            source_ontology_file,
            "-t",
            target_ontology_file,
            "-o",
            output_dir,
            "-C",
            config_file,
            "-c",
            candidates_file,
            "-r",
            reference_file,
        ],
        check=True,
    )

    print(result)


if __name__ == "__main__":
    main()
