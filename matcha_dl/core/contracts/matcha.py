from pathlib import Path
import os
import subprocess
from abc import abstractmethod

MATCHA = 'matcha'

class IMatcha:
    def __init__(self, output_file: str = 'matcha_scores.csv') -> None:
        """
        Initialize Matcher.

        Args:
            output_file (str): The path to the output file. Defaults to 'matcha_scores.csv'.
        """
        self.output_file = Path(output_file)

    @abstractmethod
    @property
    def matcha_path(self) -> Path:
        """
        Get the path to the matcha directory.

        Returns:
            Path: The path to the matcha directory.
        """
        pass
    
    @abstractmethod
    @property
    def jar_path(self) -> Path:
        """
        Get the path to the matcha.jar file.

        Returns:
            Path: The path to the matcha.jar file.
        """
        pass

    @property
    def has_cache(self) -> bool:
        """
        Check if the output file exists.

        Returns:
            bool: True if the output file exists, False otherwise.
        """
        return self.output_file.is_file()

    def match(self, ont1: str, ont2: str) -> Path:
        """
        Match two ontologies and write the result to the output file.

        Args:
            ont1 (str): The path to the first ontology.
            ont2 (str): The path to the second ontology.

        Returns:
            Path: The path to the output file.
        """
        if not self.has_cache:

            current_cwd = os.getcwd()
            os.chdir(self.matcha_path)

            subprocess.call(['java', '-jar', str(self.jar_path), str(ont1), str(ont2), str(self.output_file)])

            os.chdir(current_cwd)

        return self.output_file