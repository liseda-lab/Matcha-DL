import logging
from pathlib import Path
from typing import Optional, Protocol

from deeponto import init_jvm

from matcha_dl.impl.matcha import Matcha
from matcha_dl.impl.negative_sampler import RandomNegativeSampler
from matcha_dl.impl.processor import MainProcessor
from matcha_dl.impl.trainer import MLPTrainer
from matcha_dl.core.entities.configs import ConfigModel

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

        # Load Configs

        if configs_file_path:
            configs = ConfigModel.load_config(configs_file_path)

        else:
            logging.info(f"Loading default configuration")
            configs = ConfigModel()

        # Loading logging configuration from configs

        logging.basicConfig(
            filename=str(Path(output_dir_path) / "matcha_dl.log"),
            level=configs.logging_level,
        )

        logging.info(f"Logging level set to {configs.logging_level}")

        if configs_file_path:
            logging.info(f"Using configuration from {configs_file_path}")
        else:
            logging.info(f"Using default configuration")

        # Load JVM

        init_jvm(configs.matcha_params.max_heap)

        # Matcha module

        logging.info(f"Matching {source_file_path} and {target_file_path}")

        matcha = Matcha(
            output_file=str(Path(output_dir_path) / 'matcha_scores.csv') , 
            log_file=str(Path(output_dir_path) / 'matcha.log') ,
            **configs.matcha_params.model_dump()
        )

        logging.info(f"Computing matcha scores")
        logging.info(f"Matcha logs are being written to {matcha.log_file}")

        matcha_output_file = str(matcha.match(source_file_path, target_file_path))

        logging.info(f"Matcha scores written to {matcha_output_file}")

        # Processor module

        logging.info(f"Processing dataset")
        processor = MainProcessor(
            sampler=RandomNegativeSampler(n_samples=configs.number_of_negatives, seed=configs.seed),
            seed=configs.seed
        )

        dataset = processor.process(
            matcha_output_file, 
            reference_file_path, 
            candidates_file_path,
            output_file=str(Path(output_dir_path) / 'processed_dataset.csv')
        )

        logging.info(f"Dataset parsed")

        # Trainer module

        ## Parse model params

        model_params = configs.model_params.params
        model_params['n'] = dataset.x().shape[1]
        model_params['n_classes'] = dataset.y().shape[1]

        ## Train Model

        # TODO add has cache method to trainer that loads last checkpoint if it has any

        trainer = MLPTrainer(
            dataset=dataset,
            model=configs.model_params.model,
            loss=configs.loss_params.loss,
            optimizer=configs.optimizer_params.optimizer,
            loss_params=configs.loss_params.params,
            optimizer_params=configs.optimizer_params.params,
            model_params=model_params,
            earlystoping=None,
            device=configs.device,
            output_dir=Path(output_dir_path),
            seed=configs.seed,
            use_last_checkpoint=configs.use_last_checkpoint,
        )

        if reference_file_path:
            logging.info(f"Training model with {reference_file_path}")
            trainer.train(**configs.training_params.model_dump())

        logging.info(f"Computing alignment")

        alignment = trainer.predict(threshold=configs.threshold)

        logging.info(f"Writing alignment")

        trainer.save_alignment(alignment)

        logging.info(f"Alignment written to {trainer.alignment_dir}")
