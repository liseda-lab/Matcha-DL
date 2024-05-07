import logging
import time
from pathlib import Path
from typing import Optional, Protocol

from deeponto import init_jvm

from matcha_dl.core.entities.configs import ConfigModel
from matcha_dl.core.values import N_CLASSES
from matcha_dl.impl.matcha import Matcha
from matcha_dl.impl.negative_sampler import RandomNegativeSampler
from matcha_dl.impl.processor import MainProcessor
from matcha_dl.impl.trainer import MLPTrainer


class AlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: str,
        target_file_path: str,
        output_dir_path: str,
        configs_file_path: Optional[str] = None,
        reference_file_path: Optional[str] = None,
        candidates_file_path: Optional[str] = None,
    ) -> None:

        start_time = time.time()

        # Load Configs

        if configs_file_path is not None:
            configs = ConfigModel.load_config(configs_file_path)

        else:
            configs = ConfigModel()

        # Loading logging configuration from configs

        logger = logging.getLogger("matcha-dl")
        logger.setLevel(configs.logging_level)

        logger.debug(f"Logging level set to {configs.logging_level}")

        if configs_file_path is not None:
            logger.info(f"Using configuration from {configs_file_path}")
        else:
            logger.info(f"Using default configuration")

        # Load JVM

        init_jvm(configs.matcha_params.max_heap)

        # Matcha module

        logger.info(f"Matching {source_file_path} and {target_file_path}")

        matcha = Matcha(
            output_file=str(Path(output_dir_path) / "matcha_scores.csv"),
            log_file=str(Path(output_dir_path) / "matcha.log"),
            logger=logger,
            **configs.matcha_params.model_dump(),
        )

        logger.info(f"Computing matcha scores...")
        logger.debug(f"Matcha logs are being written to {matcha.log_file}")

        matcha_output_file, cache_ok = matcha.match(source_file_path, target_file_path)

        # Processor module

        logger.info(f"Processing dataset..")
        processor = MainProcessor(
            sampler=RandomNegativeSampler(n_samples=configs.number_of_negatives, seed=configs.seed),
            seed=configs.seed,
            logger=logger,
            cache_ok=cache_ok,
        )

        dataset = processor.process(
            matcha_output_file,
            reference_file_path,
            candidates_file_path,
            output_file=str(Path(output_dir_path) / "processed_dataset.csv"),
        )

        logger.info(f"Dataset parsed")

        # Trainer module

        ## Parse model params

        if reference_file_path is not None:

            model_params = configs.model.params
            model_params["n"] = dataset.x().shape[1]
            model_params["n_classes"] = N_CLASSES

        else:
            model_params = configs.model.params

        ## Train Model

        trainer = MLPTrainer(
            dataset=dataset,
            model=configs.model.model,
            loss=configs.loss.loss,
            optimizer=configs.optimizer.optimizer,
            loss_params=configs.loss.params,
            optimizer_params=configs.optimizer.params,
            model_params=model_params,
            earlystoping=None,
            device=configs.device,
            output_dir=Path(output_dir_path),
            seed=configs.seed,
            use_last_checkpoint=configs.use_last_checkpoint,
            logger=logger,
        )

        if reference_file_path is not None:
            logger.info(f"Training model with {reference_file_path}")
            trainer.train(**configs.training_params.model_dump())

        logger.info(f"Computing alignment...")

        alignment = trainer.predict(threshold=configs.threshold)

        logger.info(f"Writing alignment...")

        trainer.save_alignment(alignment)

        logger.info(f"Alignment written to {trainer.alignment_dir}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Alignment completed in {elapsed_time} seconds")
