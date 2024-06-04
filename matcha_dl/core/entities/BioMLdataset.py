from pathlib import Path

class BioMLDataset:
  def __init__(self, main_dir: str):

    _main_dir = Path(main_dir)
    _refs_dir = _main_dir / 'refs_equiv'
    _task = _main_dir.name
    

@property
task
source -> first element of task (plus whats after dot if there is one)
target -> second element of task (plus whats after dot if there is one)
reference -> train.tsv
candidates -> test.cands.tsv
full_reference -> full.tsv
