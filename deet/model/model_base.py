from abc import ABC, abstractmethod
import os
from collections import Counter

import pandas as pd
import numpy as np
import torch
import cv2


class DeetModelBase(ABC):
    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def set_weights(self, weights: list[np.ndarray]) -> None:
        pass

    @abstractmethod
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        pass

    @abstractmethod
    def calculate_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> int:
        pass

    def test_validate(self, classes: list, labels_files: str) -> int:
        test_images_filenames = os.listdir(os.path.join("data", "test", "images"))
        labels = pd.read_csv(os.path.join("data", "test", "images", labels_files))
        results = Counter()
        for image_filename in test_images_filenames:
            if image_filename[-4:] == ".csv":
                continue

            image = cv2.imread(os.path.join("data", "test", "images", image_filename))
            class_real = labels.at[image_filename, "class"]
            age_prediction = self.predict(image)
            results[abs(class_real - age_prediction)] += 1
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # top_left = (0, 23)
            # font_scale = 1
            # font_color = (0, 255, 0)
            # thickness = 2
            # line_type = 2
            # cv2.putText(
            #     image,
            #     str(classes[age_prediction]),
            #     top_left,
            #     font,
            #     font_scale,
            #     font_color,
            #     thickness,
            #     line_type,
            # )
            # cv2.imwrite(
            #     os.path.join(
            #         "results", f"{self._classification_task}_estimation_test", image_filename
            #     ), image
            # )

        return results[0]