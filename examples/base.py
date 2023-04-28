import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import os_lib, converter
from cv_data_parse.base import DataRegister
from pathlib import Path
from metrics import classifier, object_detection
import cv2
from cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply

MODEL = 1
WEIGHT = 2
JIT = 3
TRITON = 4


class ClsTrainDataset(Dataset):
    def __init__(self, data, augment_func=None):
        self.data = data
        self.augment_func = augment_func

    def __getitem__(self, idx):
        ret = self.data[idx]

        ret = self.aug(ret)

        image, _class = ret['image'], ret['_class']

        # (h, w, c) -> (c, w, h)
        image = np.transpose(image, (2, 1, 0))
        image = image / 255

        return torch.Tensor(image), _class

    def aug(self, ret):
        if isinstance(ret['image'], str):
            ret['image'] = cv2.imread(ret['image'])

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return len(self.data)


class OdTrainDataset(ClsTrainDataset):
    def __getitem__(self, idx):
        ret = self.data[idx]
        ret = self.aug(ret)

        image, bboxes, classes = ret['image'], ret['bboxes'], ret['classes']

        # (h, w, c) -> (c, w, h)
        image = np.transpose(image, (2, 1, 0))
        image = image / 255

        w, h = image.shape[-2:]
        img_size = np.array((w, h, w, h))

        bboxes /= img_size

        return torch.Tensor(image), torch.Tensor(bboxes), torch.Tensor(classes)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        image, bboxes, classes = zip(*batch)
        return torch.stack(image, 0), list(bboxes), list(classes)


class Process:
    train_dataset = ClsTrainDataset
    test_dataset = ClsTrainDataset

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

    @staticmethod
    def setup_seed(seed=42):
        """42 is lucky number"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=16):
        # cache_file = f'cache/{self.model_version}_train.pkl'
        # if os.path.exists(cache_file):
        #     data = os_lib.loader.load_pkl(cache_file)
        # else:
        #     data = self.get_train_data()
        #     os_lib.mk_dir('cache')
        #     os_lib.saver.save_pkl(data, cache_file)

        data = self.get_train_data()

        dataset = self.train_dataset(
            data,
            augment_func=self.data_augment
        )

        self.fit(dataset, max_epoch, train_batch_size)
        self.save(self.model_path)

        # self.load(self.model_name)

        # cache_file = f'cache/{self.model_version}_test.pkl'
        # if os.path.exists(cache_file):
        #     data = os_lib.loader.load_pkl(cache_file)
        # else:
        #     data = self.get_val_data()
        #     os_lib.mk_dir('cache')
        #     os_lib.saver.save_pkl(data, cache_file)

        data = self.get_val_data()

        dataset = self.test_dataset(data)
        print(self.metric(dataset, predict_batch_size))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def metric(self, *args, **kwargs):
        raise NotImplementedError

    def data_augment(self, x):
        return x

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_val_data(self, *args, **kwargs):
        raise NotImplementedError

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


class ClsProcess(Process):
    train_dataset = ClsTrainDataset
    test_dataset = ClsTrainDataset

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

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(crop.Random()(**ret))
        ret.update(RandomApply([
                geometry.HFlip(),
                geometry.VFlip(),
            ])(**ret))
        return ret

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


class OdProcess(Process):
    train_dataset = OdTrainDataset
    test_dataset = OdTrainDataset

    def get_train_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, )[0]

        return data

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(crop.Random()(**ret))
        # x = RandomApply([
        #     geometry.HFlip(),
        #     geometry.VFlip(),
        # ])(x)['image']
        return ret

    def get_val_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, )[0]

        return data

    def fit(self, dataset, max_epoch, batch_size):
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=8
        )

        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters())

        # 训练
        self.model.train()  # 训练模式

        for i in range(max_epoch):
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for image, bboxes, classes in pbar:
                image = image.to(self.device, non_blocking=True)

                for _ in range(len(bboxes)):
                    bboxes[_] = bboxes[_].to(device=self.device, non_blocking=True)
                    classes[_] = classes[_].to(dtype=torch.int64, device=self.device, non_blocking=True)

                optimizer.zero_grad()

                detections, det_cls, loss = self.model(image, bboxes, classes)

                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item(): .06}'})

    def predict(self, dataset, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size)  # 单卡shuffle=True，多卡shuffle=False

        pred = []
        true = []
        true_class = []
        pred_class = []

        # 预测
        with torch.no_grad():
            self.model.eval()  # 预测模式
            # 批量预测
            for image, bboxes, classes in tqdm(dataloader):  # Dataset的__getitem__返回的参数
                # 生成数据
                image = image.to(self.device, non_blocking=True)

                for _ in range(len(bboxes)):
                    bboxes[_] = bboxes[_].cpu().detach().numpy()
                    classes[_] = classes[_].cpu().detach().numpy()

                # 前向传递
                detections, cls = self.model(image)
                detections = detections.cpu().detach().numpy()
                cls = cls.cpu().detach().numpy()
                cls = np.argmax(cls, axis=1)

                true.extend(bboxes.tolist())
                pred.extend(detections.tolist())
                true_class.extend(classes.tolist())
                pred_class.extend(cls.tolist())

        pred = np.array(pred)
        true = np.array(true)
        true_class = np.array(true_class)
        pred_class = np.array(pred_class)

        return true, pred, true_class, pred_class

    def metric(self, dataset, batch_size=128):
        true, pred, true_class, pred_class = self.predict(dataset, batch_size)

        result = object_detection.AP.mAP(true, pred, classes=[true_class, pred_class])

        result.update(
            score=result['ap']
        )

        return result
