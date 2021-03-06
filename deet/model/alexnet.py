import albumentations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from deet.model.model_base import DeetModelBase

DEVICE = "cuda"
MODEL_IN_WIDTH = 224
MODEL_IN_HEIGHT = 224


class DeetAlexnet(DeetModelBase):
    def __init__(self) -> None:
        self._model = None

        self._initialize_model()

    def get_weights(self) -> list[np.ndarray]:
        weights = []
        for weights_set in self._model.parameters():
            weights.append(weights_set.data.detach().cpu().numpy())

        return weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        for parameter, weights_set in zip(self._model.parameters(), weights):
            weights_set = weights_set.astype(np.float32)
            parameter.data = torch.from_numpy(weights_set).to(DEVICE)

    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        y = y.to(DEVICE)
        x = x.to(DEVICE)
        output = self._model(x.view(-1, 3, MODEL_IN_WIDTH, MODEL_IN_HEIGHT))[0]
        loss = functional.binary_cross_entropy_with_logits(output, y).item()
        # loss_cr_entropy = functional.cross_entropy(output, y).item()
        return loss

    def calculate_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        y = y.to(DEVICE)
        x = x.to(DEVICE)
        output = self._model(x.view(-1, 3, MODEL_IN_WIDTH, MODEL_IN_HEIGHT))
        _, predictions = torch.max(output, 1)
        y = y.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions),
            "recall": recall_score(y, predictions),
            "f1": f1_score(y, predictions),
        }

    def predict(self):
        pass

    def _initialize_model(self):
        self._model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        self._model.classifier[6] = nn.Linear(
            in_features=4096, out_features=2, bias=True
        )
        self._model.to(DEVICE)
        self._model.eval()
