import os

import cv2
import numpy as np

CLASSES = {"Broccoli": 0, "Carrot": 1}


def load_veggies_data(samples_number: dict[str, int]) -> dict[int, np.ndarray]:
    all_samples_number = (
        samples_number["train"] + samples_number["val"] + samples_number["test"]
    )
    all_class_images = {}
    for class_name, class_index in CLASSES.items():
        image_files = os.listdir(
            os.path.join("data", "veggies", "Vegetable Images", "train", class_name)
        )
        image_files = image_files[:all_samples_number]
        class_images = []
        for image_file in image_files:
            image = cv2.imread(
                os.path.join(
                    "data",
                    "veggies",
                    "Vegetable Images",
                    "train",
                    class_name,
                    image_file,
                )
            )
            class_images.append(image)

        all_class_images[class_index] = np.array(class_images)

    return all_class_images
