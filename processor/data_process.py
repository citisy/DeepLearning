import copy
import torch
from torch.utils.data import Dataset, DataLoader
from utils import os_lib
from typing import Optional


class BaseDataset(Dataset):
    augment_func: Optional

    def __init__(self, iter_data, **kwargs):
        self.iter_data = iter_data
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx):
        return self.iter_data[idx]

    def __len__(self):
        return len(self.iter_data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class BaseImgDataset(BaseDataset):
    def __init__(self, iter_data, augment_func=None, complex_augment_func=None):
        super().__init__(iter_data)
        self.augment_func = augment_func
        self.complex_augment_func = complex_augment_func
        self.loader = os_lib.Loader(verbose=False)

    def __getitem__(self, idx):
        if self.complex_augment_func:
            return self.complex_augment_func(idx, self.iter_data, self.process_one)
        else:
            return self.process_one(idx)

    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = self.loader.load_img(ret['image'])

        ret['ori_image'] = ret['image']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class IterImgDataset(BaseImgDataset):
    length: int

    def process_one(self, *args):
        ret = next(self.iter_data)
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = self.loader.load_img(ret['image'])

        ret['ori_image'] = ret['image']

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return self.length


class MixDataset(BaseDataset):
    def __init__(self, obj, **kwargs):
        super().__init__(None)
        self.datasets = []
        for data, dataset_instance in obj:
            self.datasets.append(dataset_instance(data, **kwargs))

        self.nums = [len(_) for _ in self.datasets]

    def __getitem__(self, idx):
        for n, dataset in zip(self.nums, self.datasets):
            idx -= n
            if idx < 0:
                return dataset[idx]

    def __len__(self):
        return sum(self.nums)


class DataHooks:
    train_dataset_ins = BaseImgDataset
    val_dataset_ins = BaseImgDataset
    dataset_version: str
    data_dir: str

    def get_train_dataloader(self, **dataloader_kwargs):
        train_data = self.get_train_data()
        train_data = self.train_data_preprocess(train_data)

        train_dataset = self.train_dataset_ins(
            train_data,
            augment_func=self.train_data_augment,
            complex_augment_func=self.__dict__.get('complex_data_augment')
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
            **dataloader_kwargs
        )

    def get_val_dataloader(self, **dataloader_kwargs):
        val_data = self.get_val_data()
        val_data = self.val_data_preprocess(val_data)
        val_dataset = self.val_dataset_ins(val_data, augment_func=self.val_data_augment)

        return DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn,
            **dataloader_kwargs
        )

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_val_data(self, *args, **kwargs):
        raise NotImplementedError

    def train_data_preprocess(self, iter_data):
        return iter_data

    def train_data_augment(self, ret) -> dict:
        return ret

    def val_data_preprocess(self, iter_data):
        return iter_data

    def val_data_augment(self, ret) -> dict:
        return ret

    def val_data_restore(self, ret) -> dict:
        return ret

    def gen_example_data(self, batch_size=1, input_type='image_norm', **kwargs):
        if input_type == 'image':
            return torch.randint(255, (batch_size, self.in_ch, self.input_size, self.input_size), dtype=torch.uint8, device=self.device)
        elif input_type == 'image_norm':
            return torch.rand(batch_size, self.in_ch, self.input_size, self.input_size, device=self.device)
