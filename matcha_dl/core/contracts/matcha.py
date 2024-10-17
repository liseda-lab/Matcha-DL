import logging
import os
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Optional, List
import queue
import threading
import select
import time
import pandas as pd
from ast import literal_eval
from matcha_dl.impl.dp.utils import read_table

MATCHA = "matcha"


class IMatcha:
    def __init__(
        self,
        threshold: float,
        max_heap: str,
        output_path: Path,
        matchers: List[str],
        negcardinality: int,
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
        self._negcardinality = negcardinality

        self._source = None
        self._target = None
        self._reference = None
        self._candidates = self.output_path / "candidates.tsv"
        self._negatives = self.output_path / "negatives.tsv"
        self._matcha_features = self.output_path / "matcha_features.tsv"
        self._log_file = self.output_path / "matcha_error.log"

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

    @property
    def max_heap(self) -> str:
        return self._max_heap

    @property
    def negcardinality(self) -> int:
        return self._negcardinality
    
    def load_ontologies(self, source_path: Path, target_path: Path) -> None:

        self._source = source_path
        self._target = target_path

        self.log("#Loaded Ontologies Path...", level="debug")

    def load_reference(self, file_path: Path) -> None:
        self._reference = file_path

        self.log("#Loaded Reference Path...", level="debug")

    def load_candidates(self, file_path: Path) -> None:

        def get_cands(df: pd.DataFrame) -> pd.DataFrame:

            return pd.DataFrame([
                    [source, cand, None]
                    for source, _, target_cands in df.values
                    for cand in literal_eval(target_cands)
                ], columns=["Src", "Tgt", "Score"])

        # Load One2Many candidates file
        candidates = read_table(str(file_path))
        candidates.columns = ["Src", "Tgt", "Candidates"]

        # Get One2One candidates df
        candidates = get_cands(candidates)

        # Save One2One candidates
        candidates.to_csv(self.candidates, sep="\t", index=False)

        self.log("#Loaded Candidates Path...", level="debug")

    def match(self) -> None:

        def read_output(process, output_queue, stop_event):
            while not stop_event.is_set():
                # Use select to wait for the process's stdout to be ready for reading
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    line = process.stdout.readline().strip()
                    if line:
                        output_queue.put(line)
                    else:
                        break

        def wait_for_reply(matcha_process, termination):

            # Create a queue to hold the output
            output_queue = queue.Queue()

            # Create an Event object to signal the thread to stop
            stop_event = threading.Event()

            # Start a thread to read the output
            output_thread = threading.Thread(target=read_output, args=(matcha_process, output_queue, stop_event))
            output_thread.start()

            # Wait for a response from the process
            while matcha_process.poll() is None:
                try:
                    # Get a line of output from the queue
                    line = output_queue.get(timeout=1)
                    self.log(f"[Matcha] {line}", level="debug")
                    # Check for a specific response indicating the command is done
                    if termination.lower() in line.lower():
                        break
                except queue.Empty:
                    # No output received within the timeout period
                    continue

            # Signal the output thread to stop
            stop_event.set()

            # Ensure the output thread is finished
            output_thread.join()

            if matcha_process.poll() == 0:
                self.log(f"Matcha process finished without error code unespectedly", level="debug")
            elif matcha_process.poll() is not None:
                raise RuntimeError(f"Matcha subprocess returned with error code {matcha_process.returncode} check error log at {self.log_file}")

        def comunicate_matcha_process(matcha_process, input, termination=None):
            self.log(f"Running command in Matcha: {input}", level="debug")
            matcha_process.stdin.write(input + "\n")
            matcha_process.stdin.flush()

            if termination is None:
                time.sleep(1)
                return

            wait_for_reply(matcha_process, termination)

        
        def add_matchers(matcha_process):
            command = f"Matchers {{{', '.join(self.matchers)}}}"
            comunicate_matcha_process(matcha_process, command, 'matchers set')

        def generate_negatives(matcha_process):
            command = f"Negatives {self.reference} {self.negatives} {self.negcardinality}"
            comunicate_matcha_process(matcha_process, command, 'finished generating negatives')

        def generate_candidates(matcha_process):
            command = f"Match {self.threshold} {self.candidates}"
            comunicate_matcha_process(matcha_process, command, 'finished matching')

        def generate_scores(matcha_process, pairs_file):
            command = f"Score {pairs_file} {self.matcha_features}"
            comunicate_matcha_process(matcha_process, command, 'finished calculating scores')

        if self.has_cache:

            self.log(
                f"Matcha scores already exist at {self.matcha_features}. Skipping computation.",
                level="info",
            )

            return

        if self.source is None or self.target is None:
            raise FileNotFoundError("Ontologies not loaded")
        
        # Main Matcha Execution

        current_cwd = os.getcwd()
        os.chdir(self.matcha_path)

        jar_command = [
            "java",
            "-jar",
            f"-Xmx{self.max_heap}",
            str(self.jar_path),
            '-s', str(self.source),
            '-t', str(self.target),
            '-p', sys.executable,
        ]

        try:

            # Load Matcha jar file with ontologies

            self.log(f"Running Matcha with command: {jar_command}", level="debug")

            with open(self.log_file, "w") as f:

                matcha_process = subprocess.Popen(jar_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=f, text=True)

                wait_for_reply(matcha_process, 'Matcha CLI activated')

                # Load matchers to use

                add_matchers(matcha_process)

                # If candidates exist, skip, otherwise get candidates with matcha Match command

                if not self.candidates.exists():
                    generate_candidates(matcha_process)

                    if not self.candidates.exists():
                        raise FileNotFoundError(f"Matcha failed to generate candidates at {self.candidates}")

                else:
                    self.log(f"Matcha candidates already exist at {self.candidates}. Skipping computation.", level="info")

                # If reference exists, get negatives from reference, otherwirse skip

                if self.reference is not None and self.reference.exists():
                    generate_negatives(matcha_process)
                    if not self.negatives.exists():
                        raise FileNotFoundError(f"Matcha failed to generate negatives at {self.negatives}")

                # Compile all files to generate scores into a single file (reference/negatives/candidates)

                pairs_file = self.output_path / "pairs.tsv"

                with open(pairs_file, "w") as pairs:
                    with open(self.candidates, "r") as candidates:
                        pairs.write(candidates.read())
                    if self.reference is not None and self.reference.exists():
                        with open(self.reference, "r") as reference:
                            next(reference)  # Skip header
                            pairs.write(reference.read())
                        with open(self.negatives, "r") as negatives:
                            next(negatives)  # Skip header
                            pairs.write(negatives.read())

                # Use Scores command to get scores from matcha

                generate_scores(matcha_process, pairs_file)

                if not self.matcha_features.exists():
                    raise FileNotFoundError(f"Matcha failed to generate matcha features at {self.matcha_features}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Matcha subprocess returned with error code {e.returncode}")

        finally:

            # stop matcha process
            matcha_process.terminate()

            # change back to original directory
            os.chdir(current_cwd)

        self.log(f"Matcha scores written to {self.matcha_features}", level="info")

        return
        

    def log(self, msg: str, level: Optional[str] = "info"):
        if self._logger is not None:
            getattr(self._logger, level)(msg)

        else:
            print(msg)
