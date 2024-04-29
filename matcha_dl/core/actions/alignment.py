import logging
from pathlib import Path
from typing import Protocol, Optional, Dict, Any
from pydantic import BaseModel

from matcha_dl.core.contracts.matcha import MATCHA, IMatcha
from matcha_dl.core.contracts.loss import LOSS, ILoss
from matcha_dl.core.contracts.negative_sampler import NEGATIVE_SAMPLER, INegativeSampler
from matcha_dl.core.contracts.processor import PROCESSOR, IProcessor
from matcha_dl.core.contracts.trainer import TRAINER, ITrainer
from matcha_dl.core.contracts.model import MODEL, IModel
from matcha_dl.core.contracts.stopper import STOPPER, IStopper

class AlignmentActionInput(BaseModel):
    source_file_path: Path
    target_file_path: Path
    reference_file_path: Optional[Path] = None
    candidates_file_path:Optional[Path] = None
    trainer_params: Dict[str, Any] = None


class AlignmentAction(Protocol):
    @staticmethod
    def run(
        input: AlignmentActionInput,
        matcha: IMatcha,
        processor: IProcessor,
        trainer: ITrainer,

    ) -> None:
        

        # Matcha module

        logging.info(f"Matching {input.source_file_path} and {input.target_file_path}")
        matcha_output_file = matcha.match(input.source_file_path, input.target_file_path)
        logging.info(f"Matcha scores written to {matcha_output_file}")

        # Processor module

        logging.info(f"Processing dataset")
        dataset = processor.process(matcha_output_file, input.reference_file_path, input.candidates_file_path)

        # Trainer module

        if input.reference_file_path:
            logging.info(f"Training model with {input.reference_file_path}")
            trainer.train(dataset, **input.trainer_params)

        logging.info(f"Computing alignment {input.source_file_path}")

        alignment = trainer.predict()

        logging.info(f"Writing alignment to {trainer.output_file}")

        trainer.save_alignment(alignment)

        logging.info(f"Alignment written to {trainer.output_file}")



