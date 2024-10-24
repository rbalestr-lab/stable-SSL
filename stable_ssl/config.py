from dataclasses import dataclass, field, make_dataclass
from typing import Optional, Tuple
import logging
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import random
import torch

from .utils import LARS
from .model.joint_embedding import (
    BarlowTwinsConfig,
    SimCLRConfig,
    VICRegConfig,
    WMSEConfig,
)
from .model.supervised import Supervised

from .data import DataConfig


@dataclass
class OptimConfig:
    """Configuration for the optimizer used for training the model.

    Parameters:
    -----------
    optimizer : str
        Type of optimizer to use (e.g., "AdamW", "RMSprop", "SGD", "LARS").
        Default is "LARS".
    lr : float
        Learning rate for the optimizer. Default is 1e0.
    batch_size : int, optional
        Batch size for training. Default is 256.
    epochs : int, optional
        Number of epochs to train the model. Default is 10.
    max_steps : int, optional
        Maximum number of steps per epoch. Default is -1.
    weight_decay : float
        Weight decay for the optimizer. Default is 1e-6.
    momentum : float
        Momentum for the optimizer. Default is None.
    nesterov : bool
        Whether to use Nesterov momentum. Default is False.
    betas : Tuple[float, float], optional
        Betas for the AdamW optimizer. Default is (0.9, 0.999).
    grad_max_norm : float, optional
        Maximum norm for gradient clipping. Default is None.
    """

    optimizer: str = "LARS"
    lr: float = 1e0
    batch_size: int = 256
    epochs: int = 1000
    max_steps: int = -1
    weight_decay: float = 0
    momentum: Optional[float] = None
    nesterov: Optional[bool] = None
    betas: Optional[Tuple[float, float]] = None
    grad_max_norm: Optional[float] = None

    def __post_init__(self):
        """Validates and sets default values for optimizer parameters.

        Ensures that a valid optimizer is provided and assigns default values
        for parameters like learning rate, weight decay, and others, if they
        are not explicitly set. Issues warnings when defaults are used.
        """
        if not (hasattr(torch.optim, self.optimizer) or self.optimizer == "LARS"):
            raise ValueError(
                f"[stable-SSL] Invalid optimizer: {self.optimizer}. Must be a "
                "torch optimizer or 'LARS'."
            )

        # Instantiate the optimizer to get the default parameters.
        optimizer = (
            LARS if self.optimizer == "LARS" else getattr(torch.optim, self.optimizer)
        )
        default_params = optimizer([torch.tensor(0)]).defaults

        # Ensure parameters are provided appropriately based on the optimizer.
        for param in ["lr", "weight_decay", "momentum", "betas", "nesterov"]:
            if param in default_params.keys():
                if getattr(self, param) is None:
                    # If a useful parameter is not provided, its default value is used.
                    default_value = default_params[param]
                    setattr(self, param, default_value)
                    logging.warning(
                        f"[stable-SSL] {param} not provided for {self.optimizer} "
                        f"optimizer. Default value of {default_value} is used."
                    )
            else:
                # If the parameter is useless for the optimizer, it is set to None.
                setattr(self, param, None)


@dataclass
class HardwareConfig:
    """Hardware configuration for training.

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
        Number of workers for data loading. Default is 0 (data loaded in main process).
    """

    seed: Optional[int] = None
    float16: bool = False
    gpu: int = 0
    world_size: int = 1
    port: Optional[int] = None
    workers: int = 0

    def __post_init__(self):
        """Sets a random port for distributed training if not provided."""
        self.port = self.port or random.randint(49152, 65535)


@dataclass
class LogConfig:
    """Configuration for logging and checkpointing during training.

    Parameters:
    -----------
    folder : str, optional
        Path to the folder where logs and checkpoints will be saved.
        Default is the current directory + random hash folder.
    load_from : str, optional
        Path to a checkpoint from which to load the model, optimizer, and scheduler.
        Default is "ckpt".
    level : int, optional
        Logging level (e.g., logging.INFO). Default is logging.INFO.
    checkpoint_frequency : int, optional
        Frequency of saving checkpoints (in terms of epochs). Default is 10.
    save_final_model : bool, optional
        Whether to save the final trained model. Default is False.
    final_model_name : str, optional
        Name for the final saved model. Default is "final_model".
    eval_only : bool, optional
        Whether to only evaluate the model without training. Default is False.
    eval_each_epoch : bool, optional
        Whether to evaluate the model at the end of each epoch. Default is False.
    """

    folder: Optional[str] = None
    run: Optional[str] = None
    load_from: str = "ckpt"
    level: int = logging.INFO
    checkpoint_frequency: int = 10
    save_final_model: bool = False
    final_model_name: str = "final_model"
    eval_only: bool = False
    eval_each_epoch: bool = True
    api = None

    def __post_init__(self):
        """Initializes logging folder and run settings.

        If the folder path is not specified, creates a default path under `./logs`.
        The run identifier is set using the current timestamp if not provided.
        """
        if self.folder is None:
            self.folder = Path("./logs")
        else:
            self.folder = Path(self.folder)
        if self.run is None:
            self.run = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        (self.folder / self.run).mkdir(parents=True, exist_ok=True)

    @property
    def dump_path(self):
        """Returns the full path where logs and checkpoints are stored.

        This path includes the base folder and the run identifier.
        """
        return self.folder / self.run


@dataclass
class WandbConfig(LogConfig):
    """Configuration for logging with Weights & Biases (Wandb) during training.

    Parameters:
    -----------
    entity : str, optional
        Name of the (Weights & Biases) entity. Default is None.
    project : str, optional
        Name of the (Weights & Biases) project. Default is None.
    run : str, optional
        Name of the Weights & Biases run. Default is None.
    """

    entity: Optional[str] = None
    project: Optional[str] = None


@dataclass
class BaseModelConfig:
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    model : str
        Type of model to use. Default is "Supervised".
    backbone_model : str
        Neural network architecture to use for the backbone. Default is "resnet9".
    sync_batchnorm : bool, optional
        Whether to use synchronized batch normalization. Default is False.
    memory_format : str, optional
        Memory format for tensors (e.g., "channels_last"). Default is "channels_last".
    pretrained : bool, optional
        Whether to use the torchvision pretrained weights or use random initialization.
    with_classifier : bool, optional
        Whether to keep the last layer(s) of the backbone (classifier)
        when loading the model. Default is True.
    """

    name: str = "Supervised"
    backbone_model: str = "resnet18"
    sync_batchnorm: bool = False
    memory_format: str = "channels_last"
    pretrained: bool = False
    with_classifier: bool = True


@dataclass
class TrainerConfig:
    """Global configuration for training a model.

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

    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def __repr__(self) -> str:
        return OmegaConf.to_yaml(self)

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self)


_MODEL_CONFIGS = {
    "SimCLR": SimCLRConfig,
    "Barlowtwins": BarlowTwinsConfig,
    "Supervised": BaseModelConfig,
    "VICReg": VICRegConfig,
    "WMSE": WMSEConfig,
}
_LOG_CONFIGS = {
    "Wandb": WandbConfig,
    "wandb": WandbConfig,
    "None": LogConfig,
    None: LogConfig,
}


def get_args(cfg_dict, model_class=None):

    kwargs = {
        name: value
        for name, value in cfg_dict.items()
        if name not in ["data", "optim", "model", "hardware", "log"]
    }

    model = cfg_dict.get("model", {})
    if model_class is None:
        name = model.get("name", None)
    else:
        if issubclass(model_class, Supervised):
            name = "Supervised"
    model = _MODEL_CONFIGS[name](**model)

    log = cfg_dict.get("log", {})
    log = _LOG_CONFIGS[log.get("api", None)](**log)

    args = TrainerConfig(
        data=DataConfig(**cfg_dict.get("data", {})),
        optim=OptimConfig(**cfg_dict.get("optim", {})),
        model=model,
        hardware=HardwareConfig(**cfg_dict.get("hardware", {})),
        log=log,
    )

    args.__class__ = make_dataclass(
        "TrainerConfig",
        fields=[(name, type(v), v) for name, v in kwargs.items()],
        bases=(type(args),),
    )

    return args
