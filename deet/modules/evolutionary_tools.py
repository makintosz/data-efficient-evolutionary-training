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


def crossover_weights(weights: list[list[np.ndarray]]) -> list[np.ndarray]:
    newest_set = weights[0]
    for weights_set_index in range(1, len(weights)):
        newest_set = _cross_two_weights_sets(newest_set, weights[weights_set_index])

    return newest_set


def _cross_two_weights_sets(
    first_set: list[np.ndarray], second_set: list[np.ndarray]
) -> list[np.ndarray]:
    new_weights_set = []
    for first_array, second_array in zip(first_set, second_set):
        new_array = np.zeros_like(first_array)
        if len(first_array.shape) == 4:
            for filter_index in range(len(first_array)):
                random_float = np.random.uniform(0, 1)
                if random_float < 0.5:
                    new_array[filter_index] = first_array[filter_index]

                else:
                    new_array[filter_index] = second_array[filter_index]

            new_weights_set.append(new_array)

        elif len(first_array.shape) == 2:
            new_array = np.zeros_like(first_array)
            crossover_probabilities = np.random.uniform(0, 1, first_array.shape)
            new_array[crossover_probabilities < 0.5] = first_array[
                crossover_probabilities < 0.5
            ]
            new_array[crossover_probabilities >= 0.5] = second_array[
                crossover_probabilities > 0.5
            ]
            new_weights_set.append(new_array)

        elif len(first_array.shape) == 1:
            new_array = np.zeros_like(first_array)
            crossover_probabilities = np.random.uniform(0, 1, first_array.shape)
            new_array[crossover_probabilities < 0.5] = first_array[
                crossover_probabilities < 0.5
            ]
            new_array[crossover_probabilities >= 0.5] = second_array[
                crossover_probabilities > 0.5
            ]
            new_weights_set.append(new_array)

        else:
            print(f"Unknown weight shape: {first_array.shape}")
            random_float = np.random.uniform(0, 1)
            if random_float < 0.5:
                new_weights_set.append(first_array)

            else:
                new_weights_set.append(second_array)

    return new_weights_set
