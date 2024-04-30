import logging
from pathlib import Path
from typing import Optional, Protocol

import torch.optim as optim
import yaml
from deeponto import init_jvm

from matcha_dl.impl.losses.bceloss import BCELossWeighted
from matcha_dl.impl.matcha import Matcha
from matcha_dl.impl.model import MlpClassifier
from matcha_dl.impl.negative_sampler import RandomNegativeSampler
from matcha_dl.impl.processor import MainProcessor
from matcha_dl.impl.trainer import MLPTrainer


def get_optimizer(optimizer_name, parameters, learning_rate):
    if hasattr(optim, optimizer_name):
        return getattr(optim, optimizer_name)(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")


class AlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: str,
        target_file_path: str,
        output_dir_path: str,
        configs_file_path: str,
        reference_file_path: Optional[str] = None,
        candidates_file_path: Optional[str] = None,
    ) -> None:

        # Load yaml configuration file

        with open(configs_file_path, "r") as f:
            configs = yaml.safe_load(f)

        # Loading logging configuration from configs

        logging.basicConfig(
            filename=str(Path(output_dir_path) / "matcha_dl.log"),
            level=configs.logging.level,
        )

        logging.info(f"Configuration loaded from {configs_file_path}")

        # Load JVM

        init_jvm(configs.matcha_params.max_heap)

        # Matcha module

        logging.info(f"Matching {source_file_path} and {target_file_path}")

        matcha = Matcha(**configs.matcha_params)

        matcha_output_file = str(matcha.match(source_file_path, target_file_path))

        logging.info(f"Matcha scores written to {matcha_output_file}")

        # Processor module

        logging.info(f"Processing dataset")
        processor = MainProcessor(
            sampler=RandomNegativeSampler(n_samples=configs.number_of_negatives), seed=configs.seed
        )

        dataset = processor.process(matcha_output_file, reference_file_path, candidates_file_path)

        logging.info(f"Dataset parsed")

        # Trainer module
        if hasattr(optim, configs.optimizer.name):
            optimizer = getattr(optim, configs.optimizer.name)
        else:
            raise ValueError(f"Optimizer {configs.optimizer.name} not recognized")

        trainer = MLPTrainer(
            dataset=dataset,
            model=MlpClassifier,
            loss=BCELossWeighted,
            optimizer=optimizer,
            loss_params=configs.loss_params,
            optimizer_params=configs.optimizer.params,
            model_params=configs.model_params,
            earlystoping=None,
            device=configs.device,
            output_dir=Path(output_dir_path),
            seed=configs.seed,
        )

        if reference_file_path:
            logging.info(f"Training model with {reference_file_path}")
            trainer.train(**configs.train_params)

        logging.info(f"Computing alignment")

        alignment = trainer.predict()

        logging.info(f"Writing alignment")

        trainer.save_alignment(alignment)

        logging.info(f"Alignment written to {trainer.alignment_dir}")
