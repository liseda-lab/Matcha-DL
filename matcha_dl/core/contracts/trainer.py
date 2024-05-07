from abc import abstractmethod

import torch as th
from deeponto.align.mapping import EntityMapping as DeepOntoEntityMapping
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer

from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.stopper import IStopper
from matcha_dl.core.entities.dataset import MlpDataset
from matcha_dl.impl.dp.utils import fill_anchored_scores

EntityMapping = DeepOntoEntityMapping

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

Module = TorchModule
Optimizer = TorchOptimizer

TRAINER = "trainer"


def set_seed(seed_val: Optional[int] = 888):
    """Set random seed for reproducible results"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    th.manual_seed(seed_val)
    th.cuda.manual_seed_all(seed_val)

    return seed_val


class ITrainer:

    def __init__(
        self,
        dataset: MlpDataset,
        model: Type[Module],
        loss: Type[ILoss],
        optimizer: Type[Optimizer],
        loss_params: Optional[Dict[str, Any]] = {},
        optimizer_params: Optional[Dict[str, Any]] = {},
        model_params: Optional[Dict[str, Any]] = {},
        earlystoping: Optional[IStopper] = None,
        device: Optional[int] = 0,
        output_dir: Optional[Path] = None,
        seed: Optional[int] = 42,
        use_last_checkpoint: Optional[bool] = False,
        **kwargs,
    ):

        # Load Args

        self._dataset = dataset
        self._device = device
        self._model = model(**model_params).to(self.device)
        self._optimizer = optimizer(self._model.parameters(), **optimizer_params)
        self._loss = loss(device=self.device, **loss_params)
        self._earlystoping = earlystoping

        self._output_dir = output_dir
        self._seed = set_seed(seed)

        self._epoch = 1

        # Load Kwargs

        self._logger = kwargs.get("logger")

        # Load checkpoint if exists

        if use_last_checkpoint:
            if self.checkpoints_dir.is_dir():
                self.load_checkpoint()
                self.log(f"Loaded checkpoint {self._get_last_checkpoint()}")
            else:
                self.log(f"No checkpoints found in {self.checkpoints_dir}")

        # Create output directories

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.alignment_dir.mkdir(parents=True, exist_ok=True)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def device(self) -> th.device:
        return th.device(self._device if th.cuda.is_available() else "cpu")

    @property
    def dataset(self) -> MlpDataset:
        return self._dataset

    @property
    def model(self) -> Module:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def loss(self) -> ILoss:
        return self._loss

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def earlystoping(self) -> Optional[IStopper]:
        return self._earlystoping

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def checkpoints_dir(self) -> Path:
        return (self._output_dir / "training_checkpoints").resolve()

    @property
    def logs_dir(self) -> Path:
        return (self._output_dir / "training_logs").resolve()

    @property
    def alignment_dir(self) -> Path:
        return (self._output_dir / "alignment").resolve()

    @property
    def checkpoints(self) -> List[str]:
        return [x.name for x in self.checkpoints_dir.glob("**/*") if x.is_file()]

    @abstractmethod
    def train(self, epochs: Optional[int] = 100, batch_size: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def repair(self, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, threshold: Optional[float] = 0.7, **kwargs) -> List[EntityMapping]:
        pass

    def save_alignment(self, preds: List[EntityMapping]):

        if self.dataset.candidates is not None:
            return self._save_local_alignment(preds)

        else:
            return self._save_global_alignment(preds)

    def _save_global_alignment(self, preds: List[EntityMapping]):

        # Get the best mapping for each unique source entity

        all_sources = {}
        for ent_map in preds:
            if ent_map.head not in all_sources or ent_map.score > all_sources[ent_map.head].score:
                all_sources[ent_map.head] = ent_map

        # Extract the mappings as tuples

        global_alignment = EntityMapping.as_tuples(list(all_sources.values()), with_score=True)

        # Save the global alignment

        global_dir = str(self.alignment_dir) + f"/{'src2tgt.maps'}_global.tsv"

        pd.DataFrame(global_alignment, columns=["SrcEntity", "TgtEntity", "Score"]).to_csv(
            global_dir, sep="\t", index=False
        )

    def _save_local_alignment(self, preds: List[EntityMapping]):

        ranking_results = fill_anchored_scores(self.dataset.candidates.values, preds)

        local_dir = str(self.alignment_dir) + f"/{'src2tgt.maps'}_local.tsv"

        pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
            local_dir, sep="\t", index=False
        )

        return local_dir

    def load_checkpoint(self, checkpoint: Optional[str] = "last"):

        if checkpoint == "last":
            checkpoint = "{}.pt".format(self._get_last_checkpoint())

        checkpoint = th.load((self.checkpoints_dir / checkpoint).resolve())

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
        self._loss = checkpoint["loss"]

    def save_checkpoint(self):

        checkpoint = str(self._get_last_checkpoint() + 1)

        th.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            (self.checkpoints_dir / "{}.pt".format(checkpoint)).resolve(),
        )

    def _get_last_checkpoint(self) -> int:
        try:
            res = int(sorted(self.checkpoints)[-1].split(".")[0])
        except IndexError:
            res = 0

        return res

    def log(self, msg: str, level: Optional[str] = "info"):
        if self._logger is not None:
            getattr(self._logger, level)(msg)

        else:
            print(msg)
