
import subprocess
from pathlib import Path

TEST_DATA_DIR = Path('/home/pgcotovio/Matcha-DL/teststest_data/bio-ml/ncit-doid')


def main():
    source_ontology_file=str(TEST_DATA_DIR / 'ncit.owl')
    target_ontology_file=str(TEST_DATA_DIR / 'doid.owl')
    output_dir=str(TEST_DATA_DIR / 'test_output')
    config_file=None
    reference_file=None
    candidates_file=None

    result = subprocess.run(['poetry', 'run', 'matchadl', '-s', source_ontology_file, '-t', target_ontology_file, '-o', output_dir], check=True)

    print(result)


if __name__ == '__main__':
    main()