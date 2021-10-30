from .base_dataset import BaseDataset

class CubDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        if split == "train":
            names = ["cub_train"]
        elif split == "val":
            names = ["cub_val"]
        elif split == "test":
            names = ["cub_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)