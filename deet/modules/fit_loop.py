import time

from tqdm import tqdm
from torch.utils.data import Dataset

from deet.model.model_base import DeetModelBase
from deet.modules.evolutionary_tools import crossover_weights, mutate_weights


class DeetLearning:
    def __init__(self, model: DeetModelBase) -> None:
        self._model = model
        self._current_weights = self._model.get_weights()

    def fit(
        self, data_train: Dataset, data_val: Dataset, settings: dict
    ) -> dict[str, list[float]]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "total_correct": []
        }
        x_val, y_val = data_val.get_entire_set()
        for epoch in range(settings["epochs"]):
            time_start = time.time()
            for sample_index in tqdm(range(len(data_train))):
                self._model.set_weights(self._current_weights)
                sample_initial_loss = self._model.calculate_loss(
                    data_train[sample_index][0], data_train[sample_index][1]
                )
                new_weights = self._current_weights.copy()
                sample_final_loss = sample_initial_loss + 1
                while sample_final_loss > sample_initial_loss:
                    new_weights = mutate_weights(self._current_weights.copy())
                    self._model.set_weights(new_weights)
                    sample_final_loss = self._model.calculate_loss(
                        data_train[sample_index][0], data_train[sample_index][1]
                    )

                self._current_weights = crossover_weights(
                    [
                        self._current_weights,
                        new_weights
                    ]
                )

            self._model.set_weights(self._current_weights)

            # Train loss
            loss_epoch_train = 0
            for train_samples_index in range(len(data_train)):
                loss_epoch_train += self._model.calculate_loss(
                    data_train[train_samples_index][0],
                    data_train[train_samples_index][1],
                )

            history["train_loss"].append(loss_epoch_train / len(data_train))

            # Val loss
            loss_epoch_val = 0
            for val_samples_index in range(len(data_val)):
                loss_epoch_val += self._model.calculate_loss(
                    data_val[val_samples_index][0], data_val[val_samples_index][1]
                )

            history["val_loss"].append(loss_epoch_val / len(data_val))
            # Val metrics
            validation_metrics = self._model.calculate_metrics(x_val, y_val)
            # history["val_accuracy"].append(validation_metrics["accuracy"])
            # history["val_f1"].append(validation_metrics["f1"])
            # history["val_precision"].append(validation_metrics["precision"])
            # history["val_recall"].append(validation_metrics["recall"])
            history["total_correct"].append(validation_metrics["total_correct"])
            print(f"Epoka numer {epoch} - {(time.time() - time_start):.4f} seconds")

        return history
