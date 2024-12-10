from .test import CLIPTrainer, GDINOTrainer, GLIPTrainer
from .pre_train import PRETrainer
from .trainer import CoinTrainer, CoinTrainer
from .oracle_train import OracleTrainer

__all__ = list(globals().keys())