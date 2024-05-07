import logging
import os
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple

MATCHA = "matcha"


class IMatcha:
    def __init__(
        self,
        threshold: float,
        cardinality: int,
        output_file: str = "matcha_scores.csv",
        log_file: str = "matcha.log",
        max_heap="8G",
        **kwargs,
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

        self.logger = kwargs.get("logger")

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

    def match(self, ont1: str, ont2: str) -> Tuple[str, bool]:
        """
        Match two ontologies and write the result to the output file.

        Args:
            ont1 (str): The path to the first ontology.
            ont2 (str): The path to the second ontology.

        Returns:
            Path: The path to the output file.
        """
        if self.has_cache:

            self.log(
                f"Matcha scores already exist at {self.output_file}. Skipping computation.",
                level="info",
            )

            return str(self.output_file), True

        else:

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

            self.log("Running command:" + " ".join(jar_command), level="debug")

            try:
                with open(str(self.log_file), "w") as f:
                    _ = subprocess.run(jar_command, stdout=f, stderr=f, check=True)

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Matcha subprocess returned with error code {e.returncode}")

            os.chdir(current_cwd)

            self.log(f"Matcha scores written to {self.output_file}", level="info")

            return str(self.output_file), False

    def log(self, msg: str, level: Optional[str] = "info") -> None:
        if self.logger:
            getattr(self.logger, level)(msg)

        else:
            print(msg)
