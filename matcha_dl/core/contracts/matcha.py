import os
import sys
import subprocess
from abc import abstractmethod
from pathlib import Path
import logging

MATCHA = "matcha"


class IMatcha:
    def __init__(
        self,
        threshold: float,
        cardinality: int,
        output_file: str = "matcha_scores.csv",
        log_file: str = "matcha.log",
        max_heap="8G",
    ) -> None:
        """
        Initialize Matcher.

        Args:
            threshold (float): The threshold to use for matching.
            cardinality (int): The cardinality to use for matching.
            output_file (str): The path to the output file. Defaults to 'matcha_scores.csv'.
            max_heap (str): The maximum heap size to use for the Java Virtual Machine. Defaults to '8G'.
        """

        self.threshold = threshold
        self.cardinality = cardinality
        self.output_file = Path(output_file)
        self.log_file = Path(log_file)
        self.max_heap = max_heap

    @property
    @abstractmethod
    def matcha_path(self) -> Path:
        """
        Get the path to the matcha directory.

        Returns:
            Path: The path to the matcha directory.
        """
        pass
    
    @property
    @abstractmethod
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

            jar_command = [
                "java",
                "-jar",
                f"-Xmx{self.max_heap}",
                str(self.jar_path),
                str(ont1),
                str(ont2),
                str(self.output_file),
                str(self.threshold),
                str(self.cardinality),
                "true",
                sys.executable,
            ]

            # TODO take out print statement when done debugging
            print("Running command:", ' '.join(jar_command))

            try:
                with open(str(self.log_file), 'w') as f:
                    _ = subprocess.run(jar_command, stdout=f, stderr=f, check=True)
            
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Matcha subprocess returned with error code {e.returncode}")

            os.chdir(current_cwd)

        return self.output_file
