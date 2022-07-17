import numpy as np

from deet.config.data_config import DatasetType
from deet.data_processing.deet_dataset import DeetDataset
from deet.data_processing.load_and_transform import load_and_transform_data
from deet.model.alexnet import DeetAlexnet
from deet.modules.evolutionary_tools import mutate_weights

data = load_and_transform_data("veggies")
deet_dataset = DeetDataset(data, DatasetType.TRAIN)

alexnet = DeetAlexnet()

loss = alexnet.calculate_loss(deet_dataset[0][0], deet_dataset[0][1])
print(loss)

weights = alexnet.get_weights()
new_weights = mutate_weights(weights)
alexnet.set_weights(new_weights)

loss = alexnet.calculate_loss(deet_dataset[0][0], deet_dataset[0][1])
print(loss)
