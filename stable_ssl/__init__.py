# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from .__about__ import (
    __title__,
    __summary__,
    __version__,
    __url__,
    __author__,
    __license__,
)

from .joint_embedding import (
    JointEmbedding,
    NTXEnt,
    BarlowTwinsLoss,
    VICReg,
    WMSE,
    BYOL,
)
from .supervised import Supervised

__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__url__",
    "__author__",
    "__license__",
    "get_args",
    # "OptimConfig",
    # "HardwareConfig",
    # "LogConfig",
    # "WandbConfig",
    # "BaseModelConfig",
    # "TrainerConfig",
    "JointEmbedding",
    "NTXEnt",
    "Supervised",
    "SimCLR",
    "BarlowTwinsLoss",
    "VICReg",
    "WMSE",
    "BYOL",
]
