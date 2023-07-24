import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib, converter
from .base import DataRegister, DataLoader, DataSaver, get_image, save_image
from tqdm import tqdm
from pathlib import Path


class Loader(DataLoader):
    """https://labelstud.io/

    Data structure:
        .
        ├── images
        └── label_studio.json

    """

    image_suffix = 'png'
    classes = []

    def _call(self, image_type=DataRegister.ARRAY, set_task='label_studio', **kwargs):
        with open(f'{self.data_dir}/{set_task}.json', 'r', encoding='utf8') as f:
            for js in json.load(f):
                _id = js['file_upload'].split('-', 1)[-1]
                image_path = f'{self.data_dir}/images/{_id}'
                image_path = os.path.abspath(image_path)
                image = get_image(image_path, image_type)

                bboxes = []
                classes = []
                for a in js['annotations']:
                    for r in a['result']:
                        v = r['value']
                        bboxes.append([v['x'], v['y'], v['width'], v['height']])
                        classes.append(self.classes.index(v['rectanglelabels'][0]))
                        size = (r['original_height'], r['original_width'], 3)

                bboxes = np.array(bboxes)
                bboxes /= 100
                bboxes = converter.CoordinateConvert.top_xywh2top_xyxy(bboxes, wh=(size[1], size[0]), blow_up=True)
                classes = np.array(classes)

                yield dict(
                    _id=_id,
                    image=image,
                    size=size,
                    bboxes=bboxes,
                    classes=classes
                )


class Saver(DataSaver):
    classes = []

    def __call__(self, data, set_type=DataRegister.FULL, image_type=DataRegister.PATH, **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/images')
        super().__call__(data, set_type, image_type, **kwargs)

    def _call(self, iter_data, set_type, image_type, set_task='label_studio', cls_alias=None, **kwargs):
        rets = []
        for dic in tqdm(iter_data):
            _id = dic['_id']
            image = dic['image']
            image_path = f'{self.data_dir}/images/{_id}'
            save_image(image, image_path, image_type)

            if 'size' in dic:
                size = dic['size']
            elif isinstance(image, np.ndarray):
                size = image.shape[:2]
            else:
                raise 'must be set size or make image the type of np.ndarray'

            bboxes = np.array(dic['bboxes']).reshape(-1, 4)
            bboxes = converter.CoordinateConvert.top_xyxy2top_xywh(bboxes, wh=(size[1], size[0]), blow_up=False)
            bboxes *= 100
            bboxes = bboxes.tolist()
            classes = dic['classes']
            if cls_alias:
                classes = [cls_alias[i] for i in classes]

            result = []
            for box, cls in zip(bboxes, classes):
                result.append(dict(
                    id=_id,
                    original_width=size[1],
                    original_height=size[0],
                    image_rotation=0,
                    value=dict(
                        x=box[0],
                        y=box[1],
                        width=box[2],
                        height=box[3],
                        rotation=0,
                        rectanglelabels=[cls]
                    ),
                    type='rectanglelabels'
                ))

            rets.append(dict(
                file_upload=f'-{_id}',
                data=dict(image=f'-{_id}'),
                annotations=[dict(
                    result=result
                )]
            ))

        with open(f'{self.data_dir}/{set_task}.json', 'w', encoding='utf8') as f:
            json.dump(rets, f, ensure_ascii=False)

