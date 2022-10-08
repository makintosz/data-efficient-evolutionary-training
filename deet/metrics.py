import numpy as np


def hinge_loss(real:np. ndarray, pred: np.ndarray, margin: int = 1) -> float:
    results = []
    for sample_real, sample_pred in zip(real, pred):
        correct_class = int(np.argmax(sample_real))
        sample_score = 0
        for index_score, class_score in enumerate(sample_pred):
            if index_score == correct_class:
                continue

            sample_score += np.maximum(
                0, class_score - sample_pred[correct_class] + margin
            )

        results.append(sample_score)

    return np.mean(results)[0]


def hinge_loss_single_sample(
    real:np. ndarray, pred: np.ndarray, margin: int = 1
) -> float:
    correct_class = int(np.argmax(real))
    sample_score = 0
    for index_score, class_score in enumerate(pred):
        if index_score == correct_class:
            continue

        sample_score += np.maximum(
            0, class_score - pred[correct_class] + margin
        )

    return sample_score