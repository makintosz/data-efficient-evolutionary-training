import matplotlib.pyplot as plt

from deet.config.data_config import DatasetType
from deet.data_processing.deet_dataset import DeetDataset
from deet.data_processing.load_and_transform import load_and_transform_data
from deet.model.alexnet import DeetAlexnet
from deet.modules.fit_loop import DeetLearning

data = load_and_transform_data("veggies")
dataset_train = DeetDataset(data, DatasetType.TRAIN)
dataset_val = DeetDataset(data, DatasetType.VAL)

alexnet = DeetAlexnet(out_features_number=2)

deet = DeetLearning(model=alexnet)
settings = {"epochs": 150}
history = deet.fit(data_train=dataset_train, data_val=dataset_val, settings=settings)

plt.figure(figsize=(18, 9))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.legend()
plt.savefig("train.png")

plt.figure(figsize=(18, 9))
plt.plot(history["val_accuracy"], label="accuracy")
plt.plot(history["val_f1"], label="f1")
plt.plot(history["val_precision"], label="precision")
plt.plot(history["val_recall"], label="recall")
plt.legend()
plt.savefig("train_metrics.png")
