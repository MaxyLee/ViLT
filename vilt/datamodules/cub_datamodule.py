from vilt.datasets import CubDataset
from .datamodule_base import BaseDataModule

class CubDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CubDataset

    @property
    def dataset_cls_no_false(self):
        return CubDataset

    @property
    def dataset_name(self):
        return "cub"