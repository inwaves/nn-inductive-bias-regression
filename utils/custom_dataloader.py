import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CustomDataLoader(pl.LightningDataModule):
    def __init__(self, *args):
        super().__init__()
        self.train_dataset, self.test_dataset, self.device = args

    # We're doing full-batch gradient descent, so the batch_size = n
    # num_workers here should be 4 * num_GPUs available as a rule of thumb.
    def train_dataloader(self):
        if self.device == "cuda":
            return DataLoader(self.train_dataset,
                              batch_size=len(self.train_dataset),
                              num_workers=4,
                              persistent_workers=True)
        else:
            return DataLoader(self.train_dataset,
                              batch_size=len(self.train_dataset),
                              num_workers=0)

    def test_dataloader(self):
        if self.device == "cuda":
            return DataLoader(self.test_dataset,
                              batch_size=len(self.test_dataset),
                              num_workers=4,
                              persistent_workers=True)
        else:
            return DataLoader(self.test_dataset,
                              batch_size=len(self.test_dataset),
                              num_workers=0)
