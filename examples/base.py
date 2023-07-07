import logging
import os
import copy
import sys
import cv2
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from utils import os_lib, converter, configs
from utils.visualize import ImageVisualize
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from utils.os_lib import MemoryInfo
from metrics import classifier, object_detection
from data_parse.cv_data_parse.base import DataRegister
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply
from data_parse.cv_data_parse.data_augmentation import channel

configs.logger_init()

MODEL = 1
WEIGHT = 2
ONNX = 3
JIT = 4
TRITON = 5


def setup_seed(seed=42):
    """42 is lucky number"""
    import torch.backends.cudnn as cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class ClsDataset(Dataset):
    def __init__(self, data, augment_func=None, complex_augment_func=None):
        self.data = data
        self.augment_func = augment_func
        self.complex_augment_func = complex_augment_func

    def __getitem__(self, idx):
        ret = self.process_one(idx)
        image, _class = ret['image'], ret['_class']

        return torch.Tensor(image), _class

    def process_one(self, idx):
        ret = copy.deepcopy(self.data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])

        ret['ori_image'] = ret['image']
        ret['ori_bboxes'] = ret['bboxes']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return len(self.data)


class OdDataset(ClsDataset):
    def __getitem__(self, idx):
        if self.complex_augment_func:
            return self.complex_augment_func(idx, self.data, self.process_one)
        return self.process_one(idx)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class Process:
    dataset = ClsDataset
    setup_seed()

    def __init__(self, model=None, model_version=None, dataset_version='ImageNet2012', device='1', input_size=224):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") if device is not None else 'cpu'
        self.model = model
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.model_dir = f'model_data/{self.model_version}'
        os_lib.mk_dir(self.model_dir)
        self.model_path = f'{self.model_dir}/{self.dataset_version}.pth'
        self.save_result_dir = f'cache_data/{self.dataset_version}'
        self.input_size = input_size
        self.logger = logging.getLogger()
        self.model_info()

    def model_info(self, depth=3):
        profile = ModuleInfo.profile_per_layer(self.model, depth=depth)
        s = f'module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

        for p in profile:
            s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

        self.logger.info(s)

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=None, save_period=None):
        data = self.get_train_data()

        dataset = self.dataset(
            data,
            augment_func=self.data_augment,
            complex_augment_func=self.complex_data_augment if hasattr(self, 'complex_data_augment') else None
        )

        self.fit(dataset, max_epoch, train_batch_size, save_period)
        self.save(self.model_path)

        # self.load(self.model_path)
        # self.load(f'{self.model_dir}/{self.dataset_version}_last.pth')

        data = self.get_val_data()

        dataset = self.dataset(data, augment_func=self.val_data_augment)
        r = self.metric(
            dataset, predict_batch_size or train_batch_size,
            num_workers=16
        )
        for k, v in r.items():
            self.logger.info(k)
            self.logger.info(v)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def metric(self, *args, **kwargs):
        raise NotImplementedError

    def data_augment(self, ret):
        ret.update(Apply([
            pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(Apply([
            pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_restore(self, ret):
        return ret

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_val_data(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, save_path, save_type=MODEL, verbose=True):
        os_lib.mk_dir(Path(save_path).parent)

        if save_type == MODEL:
            torch.save(self.model, save_path)
        elif save_type == WEIGHT:
            torch.save(self.model.state_dict(), save_path)
        elif save_type == JIT:
            trace_input = torch.rand(1, 3, self.input_size, self.input_size).to(self.device)
            model = Export.to_jit(self.model, trace_input)
            model.save(save_path)
        elif save_type == TRITON:
            pass
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            self.logger.info(f'Successfully saved to {save_path} !')

    def load(self, save_path, save_type=MODEL, verbose=True):
        if save_type == MODEL:
            self.model = torch.load(save_path, map_location=self.device)
        elif save_type == WEIGHT:
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        elif save_type == JIT:
            self.model = torch.jit.load(save_path, map_location=self.device)
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            self.logger.info(f'Successfully load {save_path} !')


class ClsProcess(Process):
    dataset = ClsDataset

    def fit(self, dataset, max_epoch, batch_size, save_period=None, dataloader_kwargs=dict()):
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, **dataloader_kwargs)

        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters())
        loss_func = nn.CrossEntropyLoss()
        score = -1

        self.model.train()

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

                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'cpu_info': MemoryInfo.get_process_mem_info(),
                    'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if save_period and i % save_period == save_period - 1:
                self.save(f'{self.model_dir}/{self.dataset_version}_last.pth')

                val_data = self.get_val_data()
                val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)
                result = self.metric(val_dataset, batch_size)
                if result['score'] > score:
                    self.save(f'{self.model_dir}/{self.dataset_version}_best.pth')
                    score = result['score']

    def predict(self, dataset, batch_size=128, dataloader_kwargs=dict()):
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)  # 单卡shuffle=True，多卡shuffle=False

        pred = []
        true = []
        with torch.no_grad():
            self.model.eval()
            for data, label in tqdm(dataloader):
                data = data.to(self.device)
                label = label.cpu().detach().numpy()

                p = self.model(data).cpu().detach().numpy()
                p = np.argmax(p, axis=1)

                pred.extend(p.tolist())
                true.extend(label.tolist())

        pred = np.array(pred)
        true = np.array(true)

        return true, pred

    def metric(self, dataset, batch_size=128, **kwargs):
        true, pred = self.predict(dataset, batch_size, **kwargs)

        result = classifier.top_metric.f_measure(true, pred)

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
        return super().data_augment(ret)

    def get_train_data(self):
        """example"""
        from data_parse.cv_data_parse.ImageNet import Loader

        loader = Loader(f'data/ImageNet2012')
        convert_class = {7: 0, 40: 1}

        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
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
        from data_parse.cv_data_parse.ImageNet import Loader
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
    dataset = OdDataset

    def get_train_data(self):
        """example"""
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, task='')[0]

        return data

    def data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def complex_data_augment(self, idx, data, base_process):
        return base_process(idx)

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret

    def get_val_data(self):
        """example"""
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, task='')[0]
        # data = data[:20]

        return data

    def fit(self, dataset, max_epoch, batch_size, save_period=None, **dataloader_kwargs):
        # sampler = distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            # sampler=sampler,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)

        # optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        weight_decay = 0.0005
        optimizer = optim.SGD(g[2], lr=0.01, momentum=0.937, nesterov=True)
        optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)

        del g

        lrf = 0.01

        # lf = lambda x: (1 - x / max_epoch) * (1.0 - lrf) + lrf

        # cos_lr
        lf = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = -1

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        stopper = EarlyStopping(patience=10, stdout_method=self.logger.info)

        max_score = -1
        accumulate = 64 / batch_size
        j = 0

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                gt_boxes = [torch.from_numpy(ret['bboxes']).to(self.device) for ret in rets]
                gt_cls = [torch.from_numpy(ret['classes']).to(self.device) for ret in rets]
                images = torch.stack(images)

                # note that, if the images have the same shape, minmax after stack if possible
                # it can reduce about 20 seconds per epoch to voc dataset
                images = images / 255

                # note that, amp method can make the model run in dtype of half
                # even though input has dtype of torch.half and weight has dtype of torch.float
                # so that, it would run in lower memory and cost less time
                with torch.cuda.amp.autocast(True):
                    output = self.model(images, gt_boxes, gt_cls)
                loss = output['loss']

                scaler.scale(loss).backward()
                if j % accumulate == 0:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()

                j += 1

                total_loss += loss.item()
                total_batch += len(rets)

                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'mean_loss': f'{total_loss / total_batch:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if save_period and i % save_period == save_period - 1:
                self.save(f'{self.model_dir}/{self.dataset_version}_last.pth')

                val_data = self.get_val_data()
                val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)
                result = self.metric(val_dataset, batch_size, **dataloader_kwargs)
                score = result['score']
                self.logger.info(f"epoch: {i}, score: {score}")

                if score > max_score:
                    self.save(f'{self.model_dir}/{self.dataset_version}_best.pth')
                    max_score = score

                if stopper(epoch=i, fitness=score):
                    break

        scheduler.step()

    def predict(self, dataset, batch_size=128, visualize=False, save_ret_func=None, **dataloader_kwargs):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)
        gt_rets, det_rets = [], []

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(dataloader, desc='val'):
                images = [torch.Tensor(ret['image']).to(self.device) for ret in rets]
                images = torch.stack(images)
                images = images / 255

                outputs = self.model(images)
                outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

                for i in range(len(images)):
                    output = outputs[i]
                    ret = rets[i]

                    if visualize:
                        self.visualize(ret, output, save_name=f'{self.save_result_dir}/{ret["_id"]}')

                    output = configs.merge_dict(ret, output)
                    ret = self.val_data_restore(ret)
                    output = self.val_data_restore(output)

                    gt_rets.append(dict(
                        _id=ret['_id'],
                        bboxes=ret['bboxes'],
                        classes=ret['classes'],
                    ))

                    det_rets.append(dict(
                        _id=ret['_id'],
                        bboxes=output['bboxes'],
                        classes=output['classes'],
                        confs=output['confs']
                    ))

        if save_ret_func:
            save_ret_func(det_rets)

        return gt_rets, det_rets

    def metric(self, dataset, batch_size=128, **kwargs):
        gt_rets, det_rets = self.predict(dataset, batch_size, **kwargs)
        df = object_detection.quick_metric(gt_rets, det_rets, save_path=f'{self.model_dir}/{self.dataset_version}.csv', verbose=False)

        result = dict(
            per_class_result=df,
            score=df['ap']['mean']
        )

        return result

    def visualize(self, true_ret, pred_ret, save_name=''):
        img = true_ret['ori_image']
        true_ret = self.val_data_restore(true_ret)
        img1 = ImageVisualize.label_box(img, true_ret['bboxes'], true_ret['classes'], line_thickness=2)

        pred_ret_ = true_ret.copy()
        pred_ret_['bboxes'] = pred_ret['bboxes']
        pred_ret = self.val_data_restore(pred_ret_)
        img2 = ImageVisualize.label_box(img, pred_ret['bboxes'], pred_ret['classes'], line_thickness=2)

        img = np.concatenate([img1, img2], 1)

        os_lib.mk_dir(Path(save_name).parent)
        cv2.imwrite(save_name, img)
