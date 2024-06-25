from pathlib import Path

class BioMLDataset:
  def __init__(self, main_dir: str):

    self._main_dir = Path(main_dir)

    self._refs_dir = self._main_dir / 'refs_equiv'
    self._task = self._main_dir.name

    if "." in self._task:

      tnames, extra = self._task.split(".")
      n1_base, n2_base = tnames.split("-")

      self._source = self._main_dir / Path(n1_base+"."+extra+".owl")
      self._target = self._main_dir / Path(n2_base+"."+extra+".owl")
    else:

      n1_base, n2_base = self._task.split("-")

      self._source = self._main_dir /  Path(n1_base+".owl")
      self._target = self._main_dir /  Path(n2_base+".owl")
    self._source = Path(self._source).resolve()
    self._target = Path(self._target).resolve()

    self._reference = (self._refs_dir / 'train.tsv').resolve()
    self._candidates = (self._refs_dir / 'test.cands.tsv').resolve()
    self._full_reference = (self._refs_dir / 'full.tsv').resolve()
    
@property
def task(self):
  return self._task

@property
def source(self):
  return self._source

@property
def target(self):
  return self._target
  
@property
def reference(self):
  return self._reference
  
@property
def candidates(self):
  return self._candidates

@property
def full_reference(self):
  return self._full_reference
