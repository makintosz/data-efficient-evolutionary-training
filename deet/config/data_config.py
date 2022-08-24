from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Dataset(Enum):
    VEGGIES = "veggies"
    AGE = "age"
