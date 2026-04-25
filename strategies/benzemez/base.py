from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of {-1: short/exit, 0: hold, 1: long} aligned to df.index."""

    @abstractmethod
    def get_param_space(self) -> dict:
        """
        Describe the Optuna search space.
        Format: {param_name: ('int'|'float'|'categorical', low, high)}
        """

    @abstractmethod
    def set_params(self, params: dict):
        """Apply a parameter dict from the optimizer."""

    def get_params(self) -> dict:
        return {}

    def name(self) -> str:
        return self.__class__.__name__
