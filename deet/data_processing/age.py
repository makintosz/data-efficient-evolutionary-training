import os

import cv2
import numpy as np
import pandas as pd

CLASSES = {"[1-6]": 0, "[6-23]": 1, "[24-53]": 2, "[54-100]": 3}


def load_age_data() -> dict[int, np.ndarray]:
    data_labels = pd.read_csv(os.path.join("data", "age", "labels.csv"), index_col=0)
    all_class_images = {}
    for class_name, class_index in CLASSES.items():
        class_labels = data_labels[data_labels["age"] == class_index]
        class_images = []
        for row_index, (image_index, image_data) in enumerate(class_labels.iterrows()):
            if row_index == 100:
                break

            image = cv2.imread(
                os.path.join("data", "age", "images", f"{image_data['id']}.jpg")
            )
            class_images.append(image)

        all_class_images[class_index] = np.array(class_images)

    return all_class_images
