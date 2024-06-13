from pathlib import Path
import os

class FileStructure:
    def __init__(self, path: Path):
        self.path = path

    def __getattr__(self, name):
        new_path = self.path / name
        if new_path.is_dir():
            return FileStructure(new_path)
        elif new_path.is_file():
            return new_path
        raise AttributeError(f"{name} not found in {self.path}")

class BioMLResults:
    def __init__(self, main_dir: str):
        self._main_dir = Path(main_dir)
        self._populate_structure()

    def _populate_structure(self):
        for item in self._main_dir.rglob('*'):
            rel_path = item.relative_to(self._main_dir)
            parts = rel_path.parts
            current = self
            for part in parts[:-1]:
                if not hasattr(current, part):
                    setattr(current, part, FileStructure(self._main_dir / Path(*parts[:parts.index(part) + 1])))
                current = getattr(current, part)
            if item.is_dir():
                setattr(current, parts[-1], FileStructure(item))
            else:
                setattr(current, parts[-1], item)

    def __getattr__(self, name):
        new_path = self._main_dir / name
        if new_path.is_dir():

            return FileStructure(new_path.resolve())
        elif new_path.is_file():
            return Path(new_path).resolve()
        raise AttributeError(f"{name} not found in {self._main_dir}")
