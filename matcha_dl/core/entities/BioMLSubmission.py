from pathlib import Path
import os

class BioMLResults:
    def __init__(self, main_dir: str):
        self._main_dir = Path(main_dir).resolve()

    def __getattr__(self, name):
        new_path = self._main_dir / name
        print("Getting attribute", name)
        if name == "global_scope":
            return BioMLResults(Path(self._main_dir) / Path("global"))
        elif name == "local_scope":
            return BioMLResults(Path(self._main_dir) / Path("local"))
        elif name == "alignment":
            if "local" in str(self._main_dir.resolve()):
                filename = "src2tgt.maps_local.tsv"
            elif "global" in str(self._main_dir.resolve()):
                filename = "src2tgt.maps_global.tsv"
            print("Returning", filename)
            return new_path / Path(filename)
        if new_path.is_dir():
            return BioMLResults(new_path)
        elif new_path.is_file():
            return new_path
        raise AttributeError(f"{name} not found in {self._main_dir}")