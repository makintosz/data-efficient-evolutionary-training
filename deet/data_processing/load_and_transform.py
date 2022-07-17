import numpy as np

from deet.config.data_config import Dataset
from deet.data_processing.veggies import load_veggies_data

SAMPLES_NUMBER = {"train": 10, "val": 100, "test": 100}


def load_and_transform_data(dataset_name: str) -> dict[str, dict[int, np.ndarray]]:
    if dataset_name == Dataset.VEGGIES.value:
        veggies_data = load_veggies_data(samples_number=SAMPLES_NUMBER)
        veggies_data_splitted = split_data(veggies_data, SAMPLES_NUMBER)
        return veggies_data_splitted


def split_data(
    data: dict[int, np.ndarray], samples_numbers: dict[str, int]
) -> dict[str, dict[int, np.ndarray]]:
    data_splitted = {}
    for set_type, set_samples in samples_numbers.items():
        current_counter = 0
        data_splitted[set_type] = {}
        for class_index, class_data in data.items():
            data_splitted[set_type][class_index] = class_data[
                current_counter : current_counter + set_samples
            ]

        current_counter += set_samples

    # for set_name, set_data in data_splitted.items():
    #     print(set_name)
    #     for class_name, class_data in set_data.items():
    #         print(class_name)
    #         print(class_data.shape)

    return data_splitted
