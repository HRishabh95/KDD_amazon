from abc import ABC
from utils.batch_processing import TrainBatching, EvalBatching
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class KDD_amazonDataModule(pl.LightningDataModule, ABC):
    def __init__(self, train_data, val_data, test_data, train_batch_size, batch_size,
                 model_name, train_rels, val_rels, test_rels, data_fn, topics_fn, test_topics_fn):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        self.train_batch_size = train_batch_size
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        batch_processing = TrainBatching(model_name, batch_size, train_rels, data_fn, topics_fn)
        self.train_batch_processing = batch_processing.build_batch
        batch_processing = TrainBatching(model_name, batch_size, val_rels, data_fn, topics_fn)
        self.val_batch_processing = batch_processing.build_batch
        batch_processing = EvalBatching(model_name, test_rels, test_topics_fn)
        self.eval_batch_processing = batch_processing.build_batch

    def setup(self, stage=None):
        self.train_dataset = self.train_data
        self.val_dataset = self.val_data
        self.test_dataset = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            collate_fn=self.train_batch_processing
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_batch_processing
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.eval_batch_processing
        )
