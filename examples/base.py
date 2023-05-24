import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from utils.os_lib import MemoryInfo
from utils import os_lib, converter
from utils.visualize import ImageVisualize
from cv_data_parse.base import DataRegister
from metrics import classifier, object_detection
from cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, channel, RandomApply, Apply

MODEL = 1
WEIGHT = 2
ONNX = 3
JIT = 4
TRITON = 5


class ClsDataset(Dataset):
    def __init__(self, data, augment_func=None):
        self.data = data
        self.augment_func = augment_func

    def __getitem__(self, idx):
        ret = self.data[idx].copy()
        ret = self.aug(ret)
        image, _class = ret['image'], ret['_class']

        return torch.Tensor(image), _class

    def aug(self, ret):
        if isinstance(ret['image'], str):
            ret['image'] = cv2.imread(ret['image'])

        ret['ori_image'] = ret['image']

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return len(self.data)


class OdDataset(ClsDataset):
    def __getitem__(self, idx):
        ret = self.data[idx].copy()
        ret = self.aug(ret)

        ret['idx'] = idx

        return ret

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class Process:
    dataset = ClsDataset

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

        self.setup_seed()

    @staticmethod
    def setup_seed(seed=42):
        """42 is lucky number"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=16, save_period=None):
        data = self.get_train_data()

        dataset = self.dataset(
            data,
            augment_func=self.data_augment
        )

        self.fit(dataset, max_epoch, train_batch_size, save_period)
        self.save(self.model_path)

        # self.load(self.model_path)
        self.load(f'{self.model_dir}/{self.dataset_version}_last.pth')

        data = self.get_val_data()

        dataset = self.dataset(data, augment_func=self.val_data_augment)
        print(self.metric(dataset, predict_batch_size))

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
    dataset = ClsDataset

    def fit(self, dataset, max_epoch, batch_size, save_period=None):
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

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

    def predict(self, dataset, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size)  # 单卡shuffle=True，多卡shuffle=False

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
        return super().data_augment(ret)

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
    dataset = OdDataset

    def get_train_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, task='')[0]

        return data

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(scale.LetterBox()(**ret))
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        ret.update(Apply([
            pixel_perturbation.MinMax(),
            pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(scale.LetterBox()(**ret))
        ret.update(Apply([
            pixel_perturbation.MinMax(),
            pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret

    def get_val_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, task='')[0]
        # data = data[:20]

        return data

    def fit(self, dataset, max_epoch, batch_size, save_period=None):
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            # num_workers=4
        )

        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(self.model.parameters(), 0.01)

        score = -1

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0

            for rets in pbar:
                images = [torch.Tensor(ret.pop('image')).to(self.device) for ret in rets]
                gt_boxes = [torch.Tensor(ret['bboxes']).to(self.device) for ret in rets]
                gt_cls = [torch.Tensor(ret['classes']).to(self.device, dtype=torch.int64) for ret in rets]
                images = torch.stack(images)

                output = self.model(images, gt_boxes, gt_cls)
                loss = output['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                result = self.metric(val_dataset, batch_size)
                print(result['score'])
                if result['score'] > score:
                    self.save(f'{self.model_dir}/{self.dataset_version}_best.pth')
                    score = result['score']

    def predict(self, dataset, batch_size=128):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )

        self.model.to(self.device)

        gt_boxes, det_boxes, confs, true_class, pred_class = [], [], [], [], []

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(dataloader):
                images = [torch.Tensor(ret['image']).to(self.device) for ret in rets]
                images = torch.stack(images)

                outputs = self.model(images)
                outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

                for i in range(len(images)):
                    output = outputs[i]
                    ret = rets[i]

                    # self.visualize(ret, output, save_name=f'{self.save_result_dir}/{ret["_id"]}')

                    gt_boxes.append(ret['bboxes'])
                    det_boxes.append(output['bboxes'])
                    confs.append(output['conf'].tolist())
                    true_class.append(ret['classes'])
                    pred_class.append(output['classes'])

        return gt_boxes, det_boxes, confs, true_class, pred_class

    def metric(self, dataset, batch_size=128):
        gt_boxes, det_boxes, confs, true_class, pred_class = self.predict(dataset, batch_size)

        result = object_detection.AP.mAP(gt_boxes, det_boxes, confs, classes=[true_class, pred_class])

        result = {
            'per_class': {k: {
                'ap': v['ap'],
                'n_pred': len(v['pred_class']),
                'n_true': len(v['true_class'])
            } for k, v in result.items()},
            'score': sum(r['ap'] for r in result.values()) / len(result)
        }

        return result

    def visualize(self, true_ret, pred_ret, save_name=''):
        img = true_ret.pop('image')
        true_ret = self.val_data_restore(true_ret)
        img1 = ImageVisualize.label_box(img, true_ret['bboxes'], true_ret['classes'], line_thickness=2)

        pred_ret_ = true_ret.copy()
        pred_ret_['bboxes'] = pred_ret['bboxes']
        pred_ret = self.val_data_restore(pred_ret_)
        img2 = ImageVisualize.label_box(img, pred_ret['bboxes'], pred_ret['labels'], line_thickness=2)

        img = np.concatenate([img1, img2], 1)

        os_lib.mk_dir(Path(save_name).parent)
        cv2.imwrite(save_name, img)
