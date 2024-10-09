import logging
import os
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List

MATCHA = "matcha"


class IMatcha:
    def __init__(
        self,
        threshold: float,
        max_heap: str,
        output_path: Path,
        matchers: List[str],
        **kwargs,
    ) -> None:
        """
        Initialize Matcher.

        Args:
            threshold (float): The threshold to use for matching.
            output_path (Path): The path to the output directory.
            matchers (List[str]): The list of matchers to use.
            **kwargs: Additional keyword arguments.
        """

        self._threshold = threshold
        self._max_heap = max_heap
        self._output_path = output_path / "matcha"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._matchers = matchers

        self._source = None
        self._target = None
        self._reference = None
        self._candidates = self.output_path / "candidates.tsv"
        self._negatives = self.output_path / "negatives.tsv"
        self._matcha_features = self.output_path / "matcha_features.tsv"
        self._log_file = self.output_path / "matcha.log"

        self._logger = kwargs.get("logger")
        self._cache_ok = kwargs.get("cache_ok", True)

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
    def log_file(self) -> Path:
        return self._log_file

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def matchers(self) -> List[str]:
        return self._matchers

    @property
    def source(self) -> Path:
        return self._source
    
    @property
    def target(self) -> Path:
        return self._target
    
    @property
    def reference(self) -> Path:
        return self._reference
    
    @property
    def candidates(self) -> Path:
        return self._candidates
    
    @property
    def negatives(self) -> Path:
        return self._negatives
    
    @property
    def matcha_features(self) -> Path:
        return self._matcha_features
    
    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def has_cache(self) -> bool:
        
        if self._cache_ok:
            return self.matcha_features.exists()
        return False
    
    def load_ontologies(self, source_path: Path, target_path: Path) -> None:

        self._source = source_path
        self._target = target_path

        self.log("#Loaded Ontologies Path...", level="debug")

    def load_reference(self, file_path: Path) -> None:
        self._reference = file_path

        self.log("#Loaded Reference Path...", level="debug")

    def load_candidates(self, file_path: Path) -> None:
        self._candidates = file_path

        self.log("#Loaded Candidates Path...", level="debug")

    def match(self) -> None:

        def comunicate_matcha_process(matcha_process, input):
            self.log(f"Running command in Matcha: {input}", level="debug")
            stdout, stderr = matcha_process.communicate(input=input)
            self.log_file.write(stdout)
            self.log_file.write(stderr)

        def add_matchers(matcha_process, log_file):
            command = f"Matchers {{{', '.join(self.matchers)}}}"
            comunicate_matcha_process(matcha_process, command, log_file)

        def generate_negatives(matcha_process, log_file):
            command = f"negatives {self.reference} {self.negatives}"
            comunicate_matcha_process(matcha_process, command, log_file)

        def generate_candidates(matcha_process, log_file):
            command = f"Match {self.threshold} {self.candidates}"
            comunicate_matcha_process(matcha_process, command, log_file)

        def generate_scores(matcha_process, log_file, pairs_file):
            command = f"Score {pairs_file} {self.matcha_features}"
            comunicate_matcha_process(matcha_process, command, log_file)


        if self.has_cache:

            self.log(
                f"Matcha scores already exist at {self.matcha_features}. Skipping computation.",
                level="info",
            )

            return
        
        # Main Matcha Execution

        current_cwd = os.getcwd()
        os.chdir(self.matcha_path)

        jar_command = [
            "java",
            "-jar",
            f"-Xmx{self.max_heap}",
            str(self.jar_path),
            str(self.source),
            str(self.target),
            sys.executable,
        ]

        try:
            with open(str(self.log_file), "w") as f:

                # Load Matcha jar file with ontologies

                self.log(f"Running Matcha with command: {jar_command}", level="debug")

                matcha_process = subprocess.Popen(jar_command, stdin=subprocess.PIPE, stdout=f, stderr=f, text=True)

                # Load matchers to use

                add_matchers(matcha_process, f)

                # If candidates exist, skip, otherwise get candidates with matcha Match command

                if not self.candidates.exists():
                    generate_candidates(matcha_process, f)

                # If reference exists, get negatives from reference, otherwirse skip

                if self.reference is not None and self.reference.exists():
                    generate_negatives(matcha_process, f)

                # Compile all files to generate scores into a single file (reference/negatives/candidates)

                pairs_file = self.output_path / "pairs.tsv"

                with open(pairs_file, "w") as pairs:
                    with open(self.candidates, "r") as candidates:
                        pairs.write(candidates.read())
                    if self.reference is not None and self.reference.exists():
                        with open(self.reference, "r") as reference:
                            pairs.write(reference.read())
                        with open(self.negatives, "r") as negatives:
                            pairs.write(negatives.read())

                # Use Scores command to get scores from matcha

                generate_scores(matcha_process, f, pairs_file)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Matcha subprocess returned with error code {e.returncode}")
        
        os.chdir(current_cwd)

        self.log(f"Matcha scores written to {self.matcha_features}", level="info")

        return
        

    def log(self, msg: str, level: Optional[str] = "info"):
        if self._logger is not None:
            getattr(self._logger, level)(msg)

        else:
            print(msg)
