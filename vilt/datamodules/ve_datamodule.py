from vilt.datasets import VEDataset
from .datamodule_base import BaseDataModule

class VEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VEDataset

    @property
    def dataset_name(self):
        return "ve"

    def setup(self, stage):
        super().setup(stage)

        # TODO
        train_labels = self.train_dataset.table['labels'].to_pandas().tolist()
        val_labels = self.val_dataset.table['labels'].to_pandas().tolist()

        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]
        