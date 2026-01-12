import copy
import json
import os
from typing import Optional, Iterable, List, Annotated

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from utils import os_lib


class BaseDataset(Dataset):
    def __init__(self, iter_data=None, augment_func=None, **kwargs):
        self.iter_data = iter_data
        self.augment_func = augment_func
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx):
        ret = self.iter_data[idx]
        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return len(self.iter_data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class BaseImgDataset(BaseDataset):
    complex_augment_func: Optional

    def __init__(self, iter_data, augment_func=None, complex_augment_func=None, **kwargs):
        super().__init__(iter_data, augment_func, complex_augment_func=complex_augment_func, **kwargs)
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


class IterDataset(BaseDataset):
    """input iter_data is an Iterator not a list,
    it will get repeat data in multiprocess DataLoader mode,
    or set `worker_init_fn()` specially to support multiprocess"""
    length: int  # one epoch num steps

    def __getitem__(self, idx):
        ret = next(self.iter_data)
        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return self.length


class IterImgDataset(BaseImgDataset):
    """input iter_data is an Iterator not a list
    it will get repeat data in multiprocess DataLoader mode,
    or set `worker_init_fn()` specially to support multiprocess"""
    length: int  # one epoch num steps

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


class BatchIterDataset(IterableDataset):
    """input iter_data is a type of List[Iterator]
    support multiprocess DataLoader mode"""

    def __init__(self, get_func, augment_func=None, **kwargs):
        self.get_func = get_func
        self.augment_func = augment_func
        self.__dict__.update(**kwargs)

    def __iter__(self):
        iter_data = self.get_func()
        worker_info = get_worker_info()
        if worker_info is None:
            for batch_rets in iter_data:
                for ret in batch_rets:
                    yield self.process_one(ret)
        else:
            worker_id = worker_info.id
            for i in range(0, len(iter_data), worker_info.num_workers):
                j = i + worker_id
                if j < len(iter_data):
                    for ret in iter_data[j]:
                        yield self.process_one(ret)

    def process_one(self, ret):
        ret = copy.deepcopy(ret)
        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class BatchIterImgDataset(BatchIterDataset):
    def __init__(self, iter_data, augment_func=None, **kwargs):
        super().__init__(iter_data, augment_func, **kwargs)
        self.loader = os_lib.Loader(verbose=False)

    def process_one(self, ret):
        ret = copy.deepcopy(ret)
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = self.loader.load_img(ret['image'])

        ret['ori_image'] = ret['image']

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class IterBatchDataset(BaseDataset):
    """for multiprocessing communication,
    input iter_data must be a type of `multiprocessing.Queue`,
    which can get a data with type of Iterator[Iterator]"""
    length: int  # one epoch num steps

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.caches = []

    def __getitem__(self, idx):
        if not self.caches:
            self.caches = self.iter_data.get()

        ret = self.caches.pop(0)

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return self.length


class IterIterDataset(IterableDataset):
    """for multiprocessing communication,
    input iter_data must be a type of `multiprocessing.Queue`,
    which can get a data with type of Iterator[Iterator]
    todo: Queue do not support generator"""

    length: int  # one epoch num steps

    def __init__(self, iter_data, augment_func=None, **kwargs):
        self.iter_data = iter_data
        self.augment_func = augment_func
        self.caches = []
        self.__dict__.update(**kwargs)

    def __iter__(self):
        worker_info = get_worker_info()
        n = 1 if worker_info is None else worker_info.num_workers

        i = 0
        while i < self.length:
            rets = self.iter_data.get()
            for ret in rets:
                yield self.process_one(ret)

                i += n

    def process_one(self, ret):
        ret = copy.deepcopy(ret)
        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class MixDataset(BaseDataset):
    """input more than one iter_data"""

    def __init__(self, datasets, **kwargs):
        super().__init__(None)
        self.datasets = datasets
        self.length = [len(_) for _ in self.datasets]
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx):
        for n, dataset in zip(self.length, self.datasets):
            idx -= n
            if idx < 0:
                return dataset[idx]

    def __len__(self):
        return sum(self.length)


class IterRedisDataset(BaseDataset):
    """for multiprocess with lager file"""

    def __init__(self, *args, cacher_kwargs=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        self.cacher = os_lib.RedisCacher(**cacher_kwargs)

    def __len__(self):
        return self.cacher.client.dbsize()

    def __getitem__(self, idx):
        ret = self.cacher.get_one(idx)
        ret = {
            k.decode('utf8') if isinstance(k, bytes) else k: v.decode('utf8') if isinstance(v, bytes) else v
            for k, v in ret.items()
        }
        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def preprocess(self):
        self.cacher.delete_all()
        for i, ret in enumerate(self.iter_data):
            ret = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in ret.items()}
            self.cacher.cache_one(ret, i)


class DataHooks:
    train_dataset_ins = BaseImgDataset
    val_dataset_ins = BaseImgDataset
    train_dataloader_ins = DataLoader
    val_dataloader_ins = DataLoader
    dataset_version: Annotated[
        str,
        'for work_dir and cache_dir'
    ] = ''
    data_dir: Annotated[str, 'for loading the train or test data']

    def get_train_dataloader(self, data_preprocess_kwargs=dict(), data_get_kwargs=dict(), dataloader_kwargs=dict()):
        train_data = self.get_train_data(**data_get_kwargs)
        train_data = self.train_data_preprocess(train_data, **data_preprocess_kwargs)

        if not isinstance(train_data, Dataset):
            train_dataset = self.train_dataset_ins(
                train_data,
                augment_func=self.train_data_augment,
                complex_augment_func=getattr(self, 'complex_data_augment', None)
            )
        else:
            train_dataset = train_data

        dataloader_kwargs.setdefault('shuffle', True)
        dataloader_kwargs.setdefault('pin_memory', True)

        return self.train_dataloader_ins(
            train_dataset,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
            **dataloader_kwargs
        )

    def get_val_dataloader(self, val_data=None, data_preprocess_kwargs=dict(), data_get_kwargs=dict(), dataloader_kwargs=dict()):
        if val_data is None:
            val_data = self.get_val_data(**data_get_kwargs)
        val_data = self.val_data_preprocess(val_data, **data_preprocess_kwargs)
        if not isinstance(val_data, Dataset):
            val_dataset = self.val_dataset_ins(val_data, augment_func=self.val_data_augment)
        else:
            val_dataset = val_data

        return self.val_dataloader_ins(
            val_dataset,
            collate_fn=val_dataset.collate_fn,
            **dataloader_kwargs
        )

    def get_train_data(self, *args, **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        return self.get_data(*args, train=True, **kwargs)

    def get_val_data(self, *args, **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        return self.get_data(*args, train=False, **kwargs)

    def get_data(self, *args, train=True, **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        raise NotImplementedError

    def train_data_preprocess(self, iter_data, **kwargs):
        return self.data_preprocess(iter_data, train=True, **kwargs)

    def train_data_augment(self, ret) -> dict:
        return self.data_augment(ret, train=True)

    def val_data_preprocess(self, iter_data, **kwargs):
        return self.data_preprocess(iter_data, train=False, **kwargs)

    def val_data_augment(self, ret) -> dict:
        return self.data_augment(ret, train=False)

    def val_data_restore(self, ret) -> dict:
        return ret

    def predict_data_preprocess(self, iter_data):
        return self.val_data_preprocess(iter_data)

    def predict_data_augment(self, ret) -> dict:
        return self.val_data_augment(ret)

    def predict_data_restore(self, ret) -> dict:
        return self.val_data_restore(ret)

    def data_preprocess(self, iter_data, train=True, **kwargs):
        return iter_data

    def data_augment(self, ret, train=True) -> dict:
        return ret

    def get_model_inputs(self, loop_inputs, train=True):
        return loop_inputs

    def gen_example_data(self, batch_size=1, input_type='image_norm', **kwargs):
        if input_type == 'image':
            return torch.randint(255, (batch_size, self.in_ch, self.input_size, self.input_size), dtype=torch.uint8, device=self.device)
        elif input_type == 'image_norm':
            return torch.rand(batch_size, self.in_ch, self.input_size, self.input_size, device=self.device)
        elif input_type == 'text':
            return torch.randint(self.vacab_size, (batch_size, self.seq_len), dtype=torch.int, device=self.device)

    tokenizer: Optional
    vocab_fn: Annotated[str, 'for tokenizer']
    encoder_fn: Annotated[str, 'for tokenizer']

    def set_tokenizer(self):
        pass
