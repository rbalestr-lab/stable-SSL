from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings
import logging

import torch
from torch.optim import SGD, RMSprop, AdamW

from stable_ssl.utils import LARS


@dataclass
class DataConfig:
    """
    Configuration for the data used for training the model.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the training data.
    """

    data_dir: str = "./data"


@dataclass
class ModelConfig:
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    backbone_model : str
        Neural network architecture to use for the backbone. Default is "resnet9".
    sync_batchnorm : bool, optional
        Whether to use synchronized batch normalization. Default is False.
    memory_format : str, optional
        Memory format for tensors (e.g., "channels_last"). Default is "channels_last".
    temperature : str
        Temperature parameter for the contrastive loss. Default is 0.15.
    projector : str
        Architecture of the projector head. Default is "8192-8192-8192".
    """

    backbone_model: str = "resnet9"
    sync_batchnorm: bool = False
    memory_format: str = "channels_last"
    temperature: float = 0.15
    projector: str = "8192-8192-8192"


@dataclass
class OptimConfig:
    """
    Configuration for the optimizer used for training the model.

    Parameters:
    -----------
    optimizer : str
        Type of optimizer to use (e.g., "AdamW", "RMSprop", "SGD", "LARS").
        Default is "AdamW".
    lr : float
        Learning rate for the optimizer. Default is 1e-3.
    batch_size : int, optional
        Batch size for training. Default is 2048.
    epochs : int, optional
        Number of epochs to train the model. Default is 10.
    max_steps : int, optional
        Maximum number of steps per epoch. Default is -1.
    weight_decay : float
        Weight decay for the optimizer. Default is 0.
    momentum : float
        Momentum for the optimizer. Default is None.
    nesterov : bool
        Whether to use Nesterov momentum. Default is False.
    betas : Tuple[float, float], optional
        Betas for the AdamW optimizer. Default is (0.9, 0.999).
    grad_max_norm : float, optional
        Maximum norm for gradient clipping. Default is None.
    lr_classifier : float, optional
        Learning rate for the classifier head. Default is 0.3.
    """

    optimizer: str = "AdamW"
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 10
    max_steps: int = -1
    weight_decay: float = 0
    momentum: float = None
    nesterov: bool = False
    betas: Optional[Tuple[float, float]] = None
    grad_max_norm: Optional[float] = None
    lr_classifier: float = 0.3

    default_params = {
        "SGD": SGD([torch.tensor(0)]).defaults,
        "RMSprop": RMSprop([torch.tensor(0)]).defaults,
        "AdamW": AdamW([torch.tensor(0)]).defaults,
        "LARS": LARS([torch.tensor(0)]).defaults,
    }

    def __post_init__(self):

        if self.optimizer not in ["AdamW", "RMSprop", "SGD", "LARS"]:
            raise ValueError(
                f"[stable-SSL] Invalid optimizer: {self.optimizer}. Must be one of "
                "'AdamW', 'RMSprop', 'SGD', 'LARS'."
            )

        # Ensure parameters are provided appropriately based on the optimizer.
        for param in ["lr", "weight_decay", "momentum", "betas", "nesterov"]:
            if param in self.default_params[self.optimizer].keys():
                if getattr(self, param) is None:
                    # If a useful parameter is not provided, its default value is used.
                    default_value = self.default_params[self.optimizer][param]
                    setattr(self, param, default_value)
                    warnings.warn(
                        f"[stable-SSL] {param} not provided for {self.optimizer} "
                        f"optimizer. Default value of {default_value} is used."
                    )
            else:
                # If the parameter is useless for the optimizer, it is set to None.
                setattr(self, param, None)


@dataclass
class HardwareConfig:
    """
    Hardware configuration for training.

    Parameters:
    -----------
    seed : int, optional
        Random seed for reproducibility. Default is None.
    float16 : bool, optional
        Whether to use mixed precision (float16) for training. Default is False.
    gpu : int, optional
        GPU device ID to use for training. Default is 0.
    world_size : int, optional
        Number of processes participating in distributed training. Default is 1.
    port : int, optional
        Port number for distributed training. Default is None.
    workers: int, optional
        Number of workers for data loading. Default is 4.
    """

    seed: Optional[int] = None
    float16: bool = False
    gpu: int = 0
    world_size: int = 1
    port: Optional[int] = None
    workers: int = 4


@dataclass
class LogConfig:
    """
    Configuration for logging and checkpointing during training.

    Parameters:
    -----------
    folder : str, optional
        Path to the folder where logs and checkpoints will be saved.
        Default is the current directory.
    add_version : bool, optional
        Whether to append a version number to the folder path. Default is False.
    load_from : str, optional
        Path to a checkpoint from which to load the model, optimizer, and scheduler. Default is None.
    log_level : int, optional
        Logging level (e.g., logging.INFO). Default is logging.INFO.
    checkpoint_frequency : int, optional
        Frequency of saving checkpoints (in terms of epochs). Default is 1.
    save_final_model : bool, optional
        Whether to save the final trained model. Default is False.
    final_model_name : str, optional
        Name for the final saved model. Default is "final_model".
    eval_only : bool, optional
        Whether to only evaluate the model without training. Default is False.
    eval_each_epoch : bool, optional
        Whether to evaluate the model at the end of each epoch. Default is False.
    """

    folder: str = "."
    add_version: bool = False
    load_from: str = "ckpt"
    log_level: int = logging.INFO
    checkpoint_frequency: int = 1
    save_final_model: bool = False
    final_model_name: str = "final_model"
    eval_only: bool = False
    eval_each_epoch: bool = False


@dataclass
class TrainerConfig:
    """
    Global configuration for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    data : DataConfig
        Data configuration.
    optim : OptimConfig
        Optimizer configuration.
    architecture : ArchitectureConfig
        Model architecture configuration.
    hardware : HardwareConfig
        Hardware configuration.
    log : LogConfig
        Logging and checkpointing configuration
    """

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    log: LogConfig = field(default_factory=LogConfig)
