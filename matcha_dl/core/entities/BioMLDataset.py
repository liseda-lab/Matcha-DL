from pathlib import Path

class BioMLDataset:
  def __init__(self, main_dir: str):

    self._main_dir = Path(main_dir)
    self._refs_dir = _main_dir / 'refs_equiv'
    self._task = _main_dir.name
    
    if "." in x:
      tnames, extra = x.split(".")
      n1_base, n2_base = tnames.split("-")
      self._source = n1_base+"."+extra+".owl"
      self._target = n2_base+"."+extra+".owl"
    else:
      n1_base, n2_base = x.split("-")
      self._source = n1_base+".owl"
      self._target = n2_base+".owl"

    self._reference = self.refs_dir / 'train.tsv'
    self._candidates = self.refs_dir / 'test.cands.tsv'
    self._full_reference = self.refs_dir / 'full.tsv'
    
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
  return self.reference
  
@property
def candidates(self):
  return self._candidates

@property
def full_reference(self):
  return self._full_reference
