import logging
from typing import Optional, Type, Union

import torch.optim as optim
from pydantic import BaseModel, Field, field_validator

from matcha_dl import config, read_yaml
from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.model import IModel
from matcha_dl.impl import losses, models


class MatchaParams(BaseModel):
    max_heap: str = Field(config["matcha_params"]["max_heap"])
    cardinality: int = Field(config["matcha_params"]["cardinality"])
    threshold: float = Field(config["matcha_params"]["threshold"])


class TrainingParams(BaseModel):
    epochs: int = Field(config["training_params"]["epochs"])
    batch_size: Optional[int] = Field(config["training_params"]["batch_size"])
    save_interval: int = Field(config["training_params"]["save_interval"])


class ModelParams(BaseModel):
    model: Type[IModel] = Field(config["model"]["name"], validate_default=True)
    params: dict = Field(config["model"]["params"])

    @field_validator("model", mode="before")
    def parse_model(model_name: str) -> IModel:
        if hasattr(models, model_name):
            return getattr(models, model_name)
        else:
            raise ValueError(f"Model {model_name} not recognized as matcha-dl model")


class LossParams(BaseModel):
    loss: Type[ILoss] = Field(config["loss"]["name"], validate_default=True)
    params: dict = Field(config["loss"]["params"])

    @field_validator("loss", mode="before")
    def parse_loss(loss_name: str) -> ILoss:
        if hasattr(losses, loss_name):
            return getattr(losses, loss_name)
        else:
            raise ValueError(f"Loss {loss_name} not recognized as matcha-dl loss")


class OptimizerParams(BaseModel):
    optimizer: Type[optim.Optimizer] = Field(config["optimizer"]["name"], validate_default=True)
    params: dict = Field(config["optimizer"]["params"])

    @field_validator("optimizer", mode="before")
    def parse_optimizer(optimizer_name: str) -> optim.Optimizer:
        if hasattr(optim, optimizer_name):
            return getattr(optim, optimizer_name)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized as torch optimizer")


class ConfigModel(BaseModel):
    number_of_negatives: int = Field(config["number_of_negatives"])
    seed: int = Field(config["seed"])
    device: Union[int, str] = Field(config["device"], validate_default=True)
    logging_level: int = Field(config["logging_level"], validate_default=True)
    use_last_checkpoint: bool = Field(config["use_last_checkpoint"])
    threshold: float = Field(config["threshold"])
    matcha_params: MatchaParams = MatchaParams()
    training_params: TrainingParams = TrainingParams()
    model: ModelParams = ModelParams()
    loss: LossParams = LossParams()
    optimizer: OptimizerParams = OptimizerParams()

    @field_validator("logging_level", mode="before")
    def parse_logging_level(logging_level: str) -> int:
        return getattr(logging, logging_level.upper())

    @field_validator("device", mode="before")
    def parse_device(cls, device: Optional[int]) -> Union[int, str]:
        if device is not None:
            return device
        else:
            return "cpu"

    @classmethod
    def load_config(cls, file_path: str) -> "ConfigModel":
        yaml_config = read_yaml(file_path)

        matcha_params = MatchaParams(**yaml_config.get("matcha_params", {}))
        training_params = TrainingParams(**yaml_config.get("training_params", {}))
        model_params = ModelParams(**yaml_config.get("model", {}))
        loss_params = LossParams(**yaml_config.get("loss", {}))
        optimizer_params = OptimizerParams(**yaml_config.get("optimizer", {}))

        # filter config for set keys
        filtered_config = {
            k: v
            for k, v in yaml_config.items()
            if v is not None
            and k in cls.model_fields
            and k not in ["matcha_params", "training_params", "model", "loss", "optimizer"]
        }

        return cls(
            matcha_params=matcha_params,
            training_params=training_params,
            model=model_params,
            loss=loss_params,
            optimize=optimizer_params,
            **filtered_config,
        )
