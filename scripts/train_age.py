import matplotlib.pyplot as plt

from deet.config.data_config import DatasetType
from deet.data_processing.deet_dataset import DeetDataset
from deet.data_processing.load_and_transform import load_and_transform_data
from deet.model.alexnet import DeetAlexnet
from deet.modules.fit_loop import DeetLearning

data = load_and_transform_data("age")
dataset_train = DeetDataset(data, DatasetType.TRAIN)
dataset_val = DeetDataset(data, DatasetType.VAL)

alexnet = DeetAlexnet(out_features_number=4)

deet = DeetLearning(model=alexnet)
settings = {"epochs": 75}
history = deet.fit(data_train=dataset_train, data_val=dataset_val, settings=settings)

plt.figure(figsize=(18, 9))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.legend()
plt.savefig("train.png")

plt.figure(figsize=(18, 9))
plt.plot(history["total_correct"], label="total correct")
plt.legend()
plt.savefig("train_metrics.png")
