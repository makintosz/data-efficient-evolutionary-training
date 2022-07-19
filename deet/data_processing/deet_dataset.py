import albumentations
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from deet.config.data_config import DatasetType


class DeetDataset(Dataset):
    def __init__(self, data: dict[str, dict[int, np.ndarray]], set_type: DatasetType):
        self._data = None
        self._transform = None
        self._y_all = None
        self._x_all = None

        self._read_data_samples(data=data, set_type=set_type)
        self._prepare_image_transform()
        self._prepare_entire_set_data()

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_one = self._data[1][item]
        x_zero = self._data[0][item]
        x_one = self._transform(image=x_one)["image"]
        x_zero = self._transform(image=x_zero)["image"]
        x = torch.stack([x_one, x_zero])
        y = np.array([1, 0]).astype(np.float32)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return self._data[1].shape[0]

    def get_entire_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x_all, self._y_all

    def _prepare_entire_set_data(self) -> None:
        x = []
        y = []
        for image in self._data[0]:
            image_tensor = self._transform(image=image)["image"]
            x.append(image_tensor)
            y.append(0)

        for image in self._data[1]:
            image_tensor = self._transform(image=image)["image"]
            x.append(image_tensor)
            y.append(1)

        x = torch.stack(x)
        y = np.array(y).reshape((-1, 1)).astype(np.float32)
        y = torch.from_numpy(y)
        self._x_all = x
        self._y_all = y

    def _read_data_samples(
        self, data: dict[str, dict[int, np.ndarray]], set_type: DatasetType
    ) -> None:
        self._data = data[set_type.value]

    def _prepare_image_transform(self) -> None:
        self._transform = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )
