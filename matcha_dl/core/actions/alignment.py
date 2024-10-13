import logging
import time
from pathlib import Path
from typing import Optional, Protocol

from deeponto import init_jvm

from matcha_dl.core.entities.configs import ConfigModel
from matcha_dl.core.values import N_CLASSES
from matcha_dl.impl.matcha import Matcha
from matcha_dl.core.entities.datasets import TabularDataset
from matcha_dl.impl.trainer import MLPTrainer


class AlignmentAction(Protocol):
    @staticmethod
    def run(
        source_file_path: Path,
        target_file_path: Path,
        output_dir_path: Path,
        configs_file_path: Optional[Path] = None,
        reference_file_path: Optional[Path] = None,
        candidates_file_path: Optional[Path] = None,
        log_file_path: Optional[Path] = None,
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

        if log_file_path is not None:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(configs.logging_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file {log_file_path}")

        logger.debug(f"Logging level set to {configs.logging_level}")

        # log configs state

        if configs_file_path is not None:
            logger.info(f"Using configuration from {configs_file_path}")
        else:
            logger.info(f"Using default configuration")

        # Load JVM

        init_jvm(configs.matcha_params.max_heap)

        # Matcha module

        logger.info(f"Matching {source_file_path} and {target_file_path}")

        matcha = Matcha(
            output_path=output_dir_path,
            logger=logger,
            **configs.matcha_params.model_dump(),
            cache_ok=configs.use_file_cache,
        )

        logger.info(f"Computing matcha scores...")
        logger.debug(f"Matcha error logs are being written to {matcha.log_file}")

        matcha.load_ontologies(source_file_path, target_file_path)

        if reference_file_path is not None:
            matcha.load_reference(reference_file_path)

        if candidates_file_path is not None:
            print(candidates_file_path)
            matcha.load_candidates(candidates_file_path)

        matcha.match()

        # Create Dataset

        logger.info(f'Building Dataset...')

        dataset = TabularDataset(
            output_path=output_dir_path,
            matchers=matcha.matchers,
            logger=logger,
            cache_ok=configs.use_file_cache,
        )

        if matcha.reference is not None:
            dataset.load_reference(matcha.reference)
            dataset.load_negatives(matcha.negatives)
        
        dataset.load_candidates(matcha.candidates)
        dataset.load_data(matcha.matcha_features)

        dataset.process()

        logger.info(f"Dataset ready")

        dataset.save()

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
            output_dir= output_dir_path / "model",
            seed=configs.seed,
            use_last_checkpoint=configs.use_last_checkpoint,
            logger=logger,
        )

        if dataset.reference is not None:
            logger.info(f"Training model with {reference_file_path}")
            trainer.train(**configs.training_params.model_dump())

        logger.info(f"Computing alignment...")

        alignment = trainer.predict(threshold=configs.threshold)

        logger.info(f"Writing alignment...")

        trainer.save_alignment(alignment, candidates_file_path)

        logger.info(f"Alignment written to {trainer.alignment_dir}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Alignment completed in {elapsed_time:.3f} seconds")
