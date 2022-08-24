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
    def __init__(self, out_features_number: int) -> None:
        self._out_features_number = out_features_number

        self._model = None

        self._initialize_model()
        self._prepare_transforms()

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
        #loss = functional.binary_cross_entropy_with_logits(output, y).item()
        loss_cr_entropy = functional.cross_entropy(output, y).item()
        return loss_cr_entropy

    def calculate_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        y = y.to(DEVICE)
        x = x.to(DEVICE)
        output = self._model(x.view(-1, 3, MODEL_IN_WIDTH, MODEL_IN_HEIGHT))
        _, predictions = torch.max(output, 1)
        #y = y.detach().cpu().numpy()
        #predictions = predictions.detach().cpu().numpy()
        return {
            "total_correct": self.test_validate([], "age_labels.csv")
            # "accuracy": accuracy_score(y, predictions),
            # "precision": precision_score(y, predictions),
            # "recall": recall_score(y, predictions),
            # "f1": f1_score(y, predictions),
        }

    def predict(self, x: np.ndarray) -> int:
        x = self._transform(image=x)["image"]
        x = x.to(DEVICE)
        output = int(
            np.argmax(
                self._model(x.view(-1, 3, MODEL_IN_WIDTH, MODEL_IN_HEIGHT)).detach().cpu().numpy()[0]
            )
        )
        return output

    def _prepare_transforms(self) -> None:
        self._transform = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )

    def _initialize_model(self):
        self._model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        self._model.classifier[6] = nn.Linear(
            in_features=4096, out_features=self._out_features_number, bias=True
        )
        self._model.to(DEVICE)
        self._model.eval()
