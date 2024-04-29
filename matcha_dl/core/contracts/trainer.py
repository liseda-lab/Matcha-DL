from abc import abstractmethod

import torch as th
from torch.nn import Module as TorchModule
from torch.optim import Optimizer as TorchOptimizer

from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.stopper import IStopper

from matcha_dl.impl.dp.utils import fill_anchored_scores

from deeponto.align.mapping import EntityMapping as DeepOntoEntityMapping

from matcha_dl.core.entities.dataset import MLPDataset

EntityMapping = DeepOntoEntityMapping

from pathlib import Path

from typing import Dict, Optional, Any, List

import random
import numpy as np

import pandas as pd

Module = TorchModule
Optimizer = TorchOptimizer

TRAINER = 'trainer'


def set_seed(seed_val: int = 888):
    """Set random seed for reproducible results
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    th.manual_seed(seed_val)
    th.cuda.manual_seed_all(seed_val)

    return seed_val

class ITrainer:

    def __init__(self,
                 model: Module,
                 loss: ILoss,
                 optimizer: Optimizer,
                 model_params: Optional[Dict[str, Any]] = {},
                 earlystoping: Optional[IStopper] = None,
                 device: Optional[int] = 0,
                 output_dir: Optional[Path] = None,
                 seed: Optional[int] = 42,
                 **kwargs
                 ):
        
        # Load Args
        
        self._dataset = None
        self._device = device
        self._model = model(**model_params).to(self.device)
        self._optimizer = optimizer(self._model.parameters())
        self._loss = loss.to(self.device)
        self._earlystoping = earlystoping

        self._output_dir = output_dir
        self._seed = set_seed(seed)

        self._epoch = 1

        # Load checkpoint if exists

        if self.checkpoint_dir.is_dir():
            self.load_checkpoint()

        # Create output directories

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    @property
    def epoch(self):
        return self._epoch

    @property
    def device(self):
        return th.device(self._device if th.cuda.is_available() else "cpu")

    @property
    def dataset(self):
        return self._dataset

    @property
    def name(self):
        return self._dataset.name

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss(self):
        return self._loss

    @property
    def seed(self):
        return self._seed

    @property
    def earlystoping(self):
        return self._earlystoping
    
    @property
    def output_dir(self):
        return self._output_dir
    
    @property
    def checkpoints_dir(self):
        return (self.output_dir / 'training_checkpoints').resolve()

    @property
    def logs_dir(self):
        return (self.output_dir / 'training_logs').resolve()

    @property
    def alignment_dir(self):
        return (self.output_dir / 'alignment').resolve()
    
    @property
    def checkpoints(self):
        return [x.name for x in self.checkpoints_dir.glob('**/*') if x.is_file()]
    
    @abstractmethod
    def train(self, dataset: MLPDataset, epochs: Optional[int] = 100, batch_size: Optional[int] = None):
        pass

    @abstractmethod
    def filter(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, dataset: Optional[MLPDataset] = None, **kwargs):
        pass

    def save_alignment(self, preds: List[EntityMapping]):

        if self.dataset.candidates:
            return self._save_local_alignment(preds)

        else:
            return self._save_global_alignment(preds)

    def _save_global_alignment(self, preds: List[EntityMapping]):

        global_res = EntityMapping.sort_entity_mappings_by_score(preds, k=1)
            
        global_res_save = [EntityMapping(src, trg, "=", score) for src, trg, score in global_res]

        global_dir = str(self.results_dir) + f"/{'src2tgt.maps'}_global.tsv"

        pd.DataFrame(global_res_save, columns=["SrcEntity", "TgtEntity", "Score"]).to_csv(global_dir, sep="\t", index=False)

    
    def _save_local_alignment(self, preds: List[EntityMapping]):

        ranking_results = fill_anchored_scores(self.dataset.candidates, preds)

        local_dir = str(self.results_dir) + f"/{'src2tgt.maps'}_local.tsv"

        pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(local_dir, sep="\t", index=False)

        return local_dir
    
    
    def load_checkpoint(self, checkpoint: Optional[str] ='last'):

        if checkpoint == 'last':
            checkpoint = '{}.pt'.format(self._get_last_checkpoint())

        checkpoint = th.load((self.checkpoint_dir / checkpoint).resolve())

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._epoch = checkpoint['epoch']
        self._loss = checkpoint['loss']

    def save_checkpoint(self):

        checkpoint = str(self._get_last_checkpoint() + 1)

        th.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, (self.checkpoint_dir / '{}.pt'.format(checkpoint)).resolve())

    def _get_last_checkpoint(self):
        try:
            res = int(sorted(self.checkpoints)[-1].split('.')[0])
        except IndexError:
            res = 0

        return res

    