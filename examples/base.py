import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import os_lib, converter
from cv_data_parse.base import DataRegister
from pathlib import Path
from metrics import classifier
import cv2
from cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply

MODEL = 1
WEIGHT = 2
JIT = 3
TRITON = 4


class TrainDataset(Dataset):
    def __init__(self, data, augment_func=None):
        self.data = data
        self.augment_func = augment_func

    def __getitem__(self, idx):
        x, y = self.data[idx]['image'], self.data[idx]['_class']

        if self.augment_func:
            x = self.augment_func(x)

        # (w, h, c) -> (c, h, w)
        x = np.transpose(x, (2, 1, 0))
        x = x / 255

        return torch.Tensor(x), y

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, data, augment_func=None):
        self.data = data
        self.augment_func = augment_func

    def __getitem__(self, idx):
        x, y = self.data[idx]['image'], self.data[idx]['_class']

        if self.augment_func:
            x = self.augment_func(x)['image']

        # (w, h, c) -> (c, h, w)
        x = np.transpose(x, (2, 1, 0))
        x = x / 255

        return torch.Tensor(x), y

    def __len__(self):
        return len(self.data)


class Process:
    def __init__(self, model=None, model_version=None, dataset_version='ImageNet2012', device='1', input_size=224):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.model_dir = f'model_data/{self.model_version}'
        os_lib.mk_dir(self.model_dir)
        self.model_path = f'{self.model_dir}/{self.dataset_version}.pth'
        self.input_size = input_size

        self.setup_seed()

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=16):
        data = self.get_train_data()

        dataset = TrainDataset(
            data,
            augment_func=self.data_augment
        )

        self.fit(dataset, max_epoch, train_batch_size)
        self.save(self.model_path)

        # self.load(self.model_name)

        data = self.get_val_data()

        dataset = TestDataset(data)
        print(self.metric(dataset, predict_batch_size))

    def fit(self, dataset, max_epoch, batch_size):

        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters())
        loss_func = nn.CrossEntropyLoss()

        # 训练
        self.model.train()  # 训练模式

        for i in range(max_epoch):
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for data, label in pbar:
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device)

                optimizer.zero_grad()

                pred = self.model(data)
                loss = loss_func(pred, label)

                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item(): .06}'})

    def save(self, save_path, save_type=MODEL, verbose=True, print_func=None):
        os_lib.mk_dir(Path(save_path).parent)

        if save_type == MODEL:
            torch.save(self.model, save_path)
        elif save_type == WEIGHT:
            torch.save(self.model.state_dict(), save_path)
        elif save_type == JIT:
            trace_input = torch.rand(1, 3, self.input_size, self.input_size).to(self.device)
            model = converter.ModelConvert.torch2jit(self.model, trace_input)
            model.save(save_path)
        elif save_type == TRITON:
            pass
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            print_func = print_func or print
            print_func(f'Successfully saved to {save_path} !')

    def load(self, save_path, save_type=MODEL, verbose=True, print_func=None):
        if save_type == MODEL:
            self.model = torch.load(save_path, map_location=self.device)
        elif save_type == WEIGHT:
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        elif save_type == JIT:
            self.model = torch.jit.load(save_path, map_location=self.device)
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            print_func = print_func or print
            print_func(f'Successfully load {save_path} !')

    def predict(self, dataset, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size)  # 单卡shuffle=True，多卡shuffle=False

        pred = []
        true = []
        # 预测
        with torch.no_grad():
            self.model.eval()  # 预测模式
            # 批量预测
            for data, label in tqdm(dataloader):  # Dataset的__getitem__返回的参数
                # 生成数据
                data = data.to(self.device)
                label = label.cpu().detach().numpy()

                # 前向传递
                p = self.model(data).cpu().detach().numpy()
                p = np.argmax(p, axis=1)

                pred.extend(p.tolist())
                true.extend(label.tolist())

        pred = np.array(pred)
        true = np.array(true)

        return true, pred

    def metric(self, dataset, batch_size=128):
        true, pred = self.predict(dataset, batch_size)

        result = classifier.TopTarget.f_measure(true, pred)

        result.update(
            score=result['f']
        )

        return result

    def data_augment(self, x):
        x = crop.Random()(x, self.input_size)['image']
        x = RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ])(x)['image']
        return x

    @staticmethod
    def setup_seed(seed=42):
        """42 is lucky number"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def get_train_data(self):
        """example"""
        from cv_data_parse.ImageNet import Loader

        loader = Loader(f'data/ImageNet2012')
        convert_class = {7: 0, 40: 1}

        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.IMAGE, generator=False,
                      wnid=[
                          'n02124075',  # Egyptian cat,
                          'n02110341'  # dalmatian, coach dog, carriage dog
                      ]
                      )[0]

        for tmp in data:
            tmp['_class'] = convert_class[tmp['_class']]

        return data

    def get_val_data(self):
        """example"""
        from cv_data_parse.ImageNet import Loader
        loader = Loader(f'data/ImageNet2012')
        convert_class = {7: 0, 40: 1}
        cache_data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False)[0]
        data = []

        for tmp in cache_data:
            if tmp['_class'] not in [7, 40]:
                continue

            tmp['_class'] = convert_class[tmp['_class']]
            x = cv2.imread(tmp['image'])
            x = scale.Proportion()(x, 256)['image']
            x = crop.Center()(x, self.input_size)['image']
            tmp['image'] = x

            data.append(tmp)

        return data
