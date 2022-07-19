from deet.config.data_config import DatasetType
from deet.data_processing.deet_dataset import DeetDataset
from deet.data_processing.load_and_transform import load_and_transform_data
from deet.model.alexnet import DeetAlexnet
from deet.modules.evolutionary_tools import crossover_weights, mutate_weights

data = load_and_transform_data("veggies")
deet_dataset = DeetDataset(data, DatasetType.TRAIN)

x, y = deet_dataset.get_entire_set()
alexnet = DeetAlexnet()

loss = alexnet.calculate_metrics(x, y)
print(loss)
exit()
weights = alexnet.get_weights()
new_weights_1 = mutate_weights(weights)
new_weights_2 = mutate_weights(weights)
new_weights_3 = mutate_weights(weights)

new_weights = crossover_weights([new_weights_1, new_weights_2, new_weights_3])
alexnet.set_weights(new_weights)

loss = alexnet.calculate_loss(deet_dataset[0][0], deet_dataset[0][1])
