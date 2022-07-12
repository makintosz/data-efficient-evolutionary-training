import numpy as np
import torchvision

from deet.model.model_base import DeetModelBase


class DeetAlexnet(DeetModelBase):
    def __init__(self) -> None:
        self._model = None

        self._initialize_model()

    def get_weights(self) -> list[np.ndarray]:
        weights = []
        for weights_set in self._model.parameters():
            weights.append(weights_set.data.numpy())

        return weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        pass

    def calculate_loss(self) -> float:
        pass

    def _initialize_model(self):
        self._model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
