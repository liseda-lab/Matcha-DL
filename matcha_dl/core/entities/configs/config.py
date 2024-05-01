

from pydantic import BaseModel, Field
from typing import Union, Optional

from matcha_dl import config, read_yaml
from matcha_dl.impl import models
from matcha_dl.impl import losses
from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.contracts.loss import ILoss

import torch.optim as optim

## Parsers

def parse_optimizer(optimizer_name: str) -> optim.Optimizer:
    if hasattr(optim, optimizer_name):
        return getattr(optim, optimizer_name)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized as torch optimizer")

def parse_device(device: Optional[int]):
    if device is not None:
        return device
    else:
        return 'cpu'

def parse_model(model_name: str) -> IModel:
    if hasattr(models, model_name):
        return getattr(models, model_name)
    else:
        raise ValueError(f"Model {model_name} not recognized as matcha-dl model")

def parse_loss(loss_name: str) -> ILoss:
    if hasattr(losses, loss_name):
        return getattr(losses, loss_name)
    else:
        raise ValueError(f"Loss {loss_name} not recognized as matcha-dl loss")

## ConfigModels

class MatchaParams(BaseModel):
    max_heap: str = Field(config['matcha_params']['max_heap'])
    cardinality: int = Field(config['matcha_params']['cardinality'])
    threshold: float = Field(config['matcha_params']['threshold'])

class TrainingParams(BaseModel):
    epochs: int = Field(config['training_params']['epochs'])
    batch_size: Optional[int] = Field(config['training_params']['batch_size'])
    save_interval: int = Field(config['training_params']['save_interval'])

class ModelParams(BaseModel):
    model: IModel = Field(config['model']['name'], pre=[parse_model])
    params: dict = Field(config['model']['params'])

class LossParams(BaseModel):
    loss: ILoss = Field(config['loss']['name'], pre=[parse_loss])
    params: dict = Field(config['loss']['params'])

class OptimizerParams(BaseModel):
    optimizer: optim.Optimizer = Field(config['optimizer']['name'], pre=[parse_optimizer])
    params: dict = Field(config['optimizer']['params'])

class ConfigModel(BaseModel):
    number_of_negatives: int = Field(config['number_of_negatives'])
    seed: int = Field(config['seed'])
    device: Union[int, str] = Field(config['device'], pre=[parse_device])
    logging_level: str = Field(config['logging_level'])
    use_last_checkpoint: bool = Field(config['use_last_checkpoint'])
    threshold: float = Field(config['threshold'])
    matcha_params: MatchaParams = MatchaParams()
    training_params: TrainingParams = TrainingParams()
    model_params: ModelParams = ModelParams()
    loss_params: LossParams = LossParams()
    optimizer_params: OptimizerParams = OptimizerParams()

    @classmethod
    def load_config(cls, file_path: str) -> 'ConfigModel':
        yaml_config = read_yaml(file_path)

        matcha_params = MatchaParams(**yaml_config.get('matcha_params', {}))
        training_params = TrainingParams(**yaml_config.get('training_params', {}))
        model_params = ModelParams(**yaml_config.get('model', {}))
        loss_params = LossParams(**yaml_config.get('loss', {}))
        optimizer_params = OptimizerParams(**yaml_config.get('optimizer', {}))

        return cls(
            number_of_negatives=yaml_config.get('number_of_negatives'),
            seed=yaml_config.get('seed'),
            device=yaml_config.get('device'),
            logging_level=yaml_config.get('logging_level'),
            use_last_checkpoint=yaml_config.get('use_last_checkpoint'),
            threshold=yaml_config.get('threshold'),
            matcha_params=matcha_params,
            training_params=training_params,
            model_params=model_params,
            loss_params=loss_params,
            optimizer_params=optimizer_params
        )