from abc import ABC, abstractmethod

import numpy as np


class DeetModelBase(ABC):
    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def set_weights(self, weights: list[np.ndarray]) -> None:
        pass

    @abstractmethod
    def calculate_loss(self) -> float:
        pass
