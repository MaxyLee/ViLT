import ipdb
from .base_dataset import BaseDataset

class VEDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        names = [f"ve_{split}"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name='text',
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, text_index = self.index_mapper[index]

        if self.split != "test":
            try:
                labels = self.table["labels"][index][text_index].as_py()
                scores = self.table["scores"][index][text_index].as_py()
            except Exception as e:
                print(f'Error: {e}')
                print(f'index: {index}')
                print(f'len1: {len(self.table["labels"])}')
                print(f'text_index: {text_index}')
                print(f'labels: {self.table["labels"][index]}')
                exit()
        else:
            labels = list()
            scores = list()

        # TODO
        return {
            "image": image_tensor,
            "text": text,
            "ve_labels": labels,
            "ve_scores": scores,
        }