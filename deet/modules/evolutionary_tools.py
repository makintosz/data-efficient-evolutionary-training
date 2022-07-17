import numpy as np

from deet.config.evolutionary_config import (
    MUTATION_DISTANCE,
    PROBABILITY_WEIGHT_MUTATION,
)


def mutate_weights(weights_data: list[np.ndarray]) -> list[np.ndarray]:
    weights_new = []
    for weights_array in weights_data:
        new_array = _mutate_array(weights_array)
        weights_new.append(new_array)

    return weights_new


def _mutate_array(array: np.ndarray, gain: float = 1) -> np.ndarray:
    values_to_add = np.random.uniform(
        -1 * MUTATION_DISTANCE * gain, MUTATION_DISTANCE * gain, array.shape
    )
    mutation_probability = np.random.uniform(0, 1, array.shape)
    weights_to_mutate = np.zeros_like(array)
    weights_to_mutate[mutation_probability < PROBABILITY_WEIGHT_MUTATION] = 1
    values_to_add *= weights_to_mutate
    return array + values_to_add
