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
from cv_data_parse.base import DataRegister
from metrics import classifier, object_detection
from cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply

MODEL = 1
WEIGHT = 2
ONNX = 3
JIT = 4
TRITON = 5


class ClsTrainDataset(Dataset):
    def __init__(self, data, augment_func=None):
        self.data = data
        self.augment_func = augment_func

    def __getitem__(self, idx):
        ret = self.data[idx].copy()

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
        ret = self.data[idx].copy()
        ret = self.aug(ret)

        image, bboxes, classes = ret['image'], ret['bboxes'], ret['classes']

        # (h, w, c) -> (c, w, h)
        image = np.transpose(image, (2, 1, 0))
        image = image / 255

        # w, h = image.shape[-2:]
        # img_size = np.array((w, h, w, h))
        # bboxes /= img_size

        ret['image'] = image

        return ret

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class Process:
    train_dataset = ClsTrainDataset
    val_dataset = ClsTrainDataset

    def __init__(self, model=None, model_version=None, dataset_version='ImageNet2012', device='1', input_size=224):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") if device is not None else 'cpu'
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

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=16, save_period=None):
        # cache file would be so big
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

        self.fit(dataset, max_epoch, train_batch_size, save_period)
        self.save(self.model_path)

        self.load(self.model_path)

        # cache file would be so big
        # cache_file = f'cache/{self.model_version}_test.pkl'
        # if os.path.exists(cache_file):
        #     data = os_lib.loader.load_pkl(cache_file)
        # else:
        #     data = self.get_val_data()
        #     os_lib.mk_dir('cache')
        #     os_lib.saver.save_pkl(data, cache_file)

        data = self.get_val_data()

        dataset = self.val_dataset(data, augment_func=self.val_data_augment)
        print(self.metric(dataset, predict_batch_size))

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def metric(self, *args, **kwargs):
        raise NotImplementedError

    def data_augment(self, ret):
        return ret

    def val_data_augment(self, ret):
        return ret

    def val_data_restore(self, *args):
        raise NotImplementedError

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
    val_dataset = ClsTrainDataset

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
                val_dataset = self.val_dataset(val_data, augment_func=self.val_data_augment)
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
    val_dataset = OdTrainDataset

    def get_train_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, )[0]

        return data

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(crop.Random()(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(scale.LetterBox()(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)

        return ret

    def get_val_data(self):
        """example"""
        from cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, )[0]

        return data

    def fit(self, dataset, max_epoch, batch_size, save_period=None):
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=4
        )

        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(self.model.parameters(), 0.01)

        self.model.train()

        for i in range(max_epoch):
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for ret in pbar:
                image, bboxes, classes = ret['image'], ret['bboxes'], ret['classes']

                for _ in range(len(image)):
                    image[_] = torch.Tensor(image[_]).to(self.device)
                    bboxes[_] = torch.Tensor(bboxes[_]).to(device=self.device)
                    classes[_] = torch.Tensor(classes[_]).to(dtype=torch.int64, device=self.device)

                image = torch.stack(image, 0)

                optimizer.zero_grad()

                detections, det_cls, loss = self.model(image, bboxes, classes)

                loss.backward()
                optimizer.step()

                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'cpu_info': MemoryInfo.get_process_mem_info(),
                    'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if save_period and i % save_period == save_period - 1:
                self.save(self.model_path)

    def predict(self, dataset, batch_size=128):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )

        gt_boxes, det_boxes, confs, true_class, pred_class = [], [], [], [], []

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(dataloader):
                image = [torch.Tensor(ret.pop('image')).to(self.device) for ret in rets]
                bboxes = [ret.pop('bboxes') for ret in rets]
                classes = [ret.pop('classes') for ret in rets]

                for _ in range(len(image)):
                    image[_] = torch.Tensor(image[_]).to(self.device, non_blocking=True)

                image = torch.stack(image, 0)

                detections, cls = self.model(image)

                for i in range(len(image)):
                    detection = detections[i].cpu().detach().numpy()

                    # from utils.visualize import ImageVisualize
                    # img = image[i].cpu().detach().numpy()
                    # img = np.transpose(img, (2, 1, 0))
                    # img = np.ascontiguousarray(img)
                    # img = img * 255
                    # img = img.astype(np.uint8)
                    # img1 = ImageVisualize.label_box(img, bboxes[i], classes[i], line_thickness=2)
                    #
                    # img2 = ImageVisualize.label_box(img, detection, cls[i], line_thickness=2)
                    # img = np.concatenate([img1, img2], 1)
                    #
                    # cv2.imwrite('test.png', img)

                    gt_ret = self.val_data_restore(dict(bboxes=bboxes[i], classes=classes[i], **rets[i]))
                    det_ret = self.val_data_restore(dict(bboxes=detection, classes=cls[i], **rets[i]))

                    gt_boxes.append(gt_ret['bboxes'].tolist())
                    det_boxes.append(det_ret['bboxes'].tolist())
                    confs.append(det_ret['classes'].tolist())
                    true_class.append(gt_ret['classes'].tolist())
                    pred_class.append(det_ret['classes'].tolist())

        gt_boxes = np.array(gt_boxes)
        det_boxes = np.array(det_boxes)
        true_class = np.array(true_class)
        pred_class = np.array(pred_class)

        return gt_boxes, det_boxes, confs, true_class, pred_class

    def metric(self, dataset, batch_size=128):
        gt_boxes, det_boxes, confs, true_class, pred_class = self.predict(dataset, batch_size)

        result = object_detection.AP.mAP(gt_boxes, det_boxes, confs, classes=[true_class, pred_class])

        result.update(
            score=result['ap']
        )

        return result
