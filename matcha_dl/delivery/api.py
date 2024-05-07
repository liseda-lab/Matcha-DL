from pathlib import Path
from typing import Optional

from matcha_dl.core.actions.alignment import AlignmentAction


class AlignmentRunner:
    """
    Main class to run the alignment.
    """

    def __init__(
        self,
        source_ontology_file: str,
        target_ontology_file: str,
        output_dir: str,
        reference_file: Optional[str] = None,
        candidates_file: Optional[str] = None,
        config_file: Optional[str] = None,
    ):
        """

        Args:
            source_ontology_file (str): Path to the source ontology file.
            target_ontology_file (str): Path to the target ontology file.
            output_dir (str): Path to the output directory.
            reference_file (str, optional): Path to the reference file. Defaults to None.
            candidates_file (str, optional): Path to the candidates file. Defaults to None.
            config_file (str, optional): Path to the configuration file. Defaults to None.
        """
        self.source_ontology_file = source_ontology_file
        self.target_ontology_file = target_ontology_file
        self.output_dir = output_dir
        self.reference_file = reference_file
        self.candidates_file = candidates_file
        self.config_file = config_file

    def run_alignment(self) -> None:

        AlignmentAction.run(
            source_file_path=self.source_ontology_file,
            target_file_path=self.target_ontology_file,
            output_dir_path=self.output_dir,
            configs_file_path=self.config_file,
            reference_file_path=self.reference_file,
            candidates_file_path=self.candidates_file,
        )

    def validate_files(self) -> None:

        if not Path(self.source_ontology_file).exists():
            raise Exception(f"Source ontology file {self.source_ontology_file} does not exist")
        if not Path(self.target_ontology_file).exists():
            raise Exception(f"Target ontology file {self.target_ontology_file} does not exist")
        if self.reference_file and not Path(self.reference_file).exists():
            raise Exception(f"Reference file {self.reference_file} does not exist")
        if self.candidates_file and not Path(self.candidates_file).exists():
            raise Exception(f"Candidates file {self.candidates_file} does not exist")
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)

        if self.config_file:
            config_file = Path(self.config_file)
            if not config_file.exists():
                raise Exception(f"Configuration file {self.config_file} does not exist")

    def run(self) -> None:

        self.validate_files()
        self.run_alignment()
