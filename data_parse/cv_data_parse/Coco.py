import os
import json
import cv2
import numpy as np
from utils import converter
from .base import DataLoader, DataRegister, get_image


class Loader(DataLoader):
    """https://cocodataset.org/#home

    Data structure(bass on Coco2017):
        .
        ├── annotations                         # include instances, captions, person_keypoints, stuff
        │   ├── instances_train2017.json		# object instances, 118287 images, 860001 annotations
        │   ├── instances_val2017.json          # 5000 images, 36781 annotations
        │   ├── captions_train2017.json			# captions
        │   ├── captions_val2017.json
        │   ├── person_keypoints_train2017.json	# person keypoints
        │   ├── person_keypoints_val2017.json
        │   ├── stuff_train2017.json			# captions
        │   └── ...
        ├── test2017
        ├── train2017
        └── val2017

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Coco import DataRegister, Loader

            loader = Loader('data/coco2017')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            bboxes = r['bboxes']
            classes = r['classes']
            classes = [loader.classes[_] for _ in classes]
            image = ImageVisualize.label_box(image, bboxes, classes, line_thickness=2)

    """
    default_set_type = [DataRegister.TRAIN, DataRegister.VAL]
    classes = None

    def _call(self, set_type, image_type, task='instances', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:
            task(str): task from annotations dir

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                size: image shape
                bboxes: a np.ndarray with shape of (-1, 4), 4 means [top_left_x, top_left_y, w, h]
                classes: a list
        """

        fn = f'{self.data_dir}/annotations/{task}_{set_type.value}2017.json'

        with open(fn, 'r', encoding='utf8') as f:
            js = json.load(f)

        if not self.classes:
            self.classes = {_['id']: _['name'] for _ in js['categories']}

        annotations = dict()
        for d in js['annotations']:
            annotations.setdefault(d['image_id'], []).append(d)

        for tmp in js['images']:
            image_path = os.path.abspath(f'{self.data_dir}/{set_type.value}2017/{tmp["file_name"]}')
            image = get_image(image_path, image_type)

            size = (tmp['width'], tmp['height'], 3)
            _id = tmp['id']

            labels = annotations.get(_id, [])

            bboxes = []
            classes = []
            for label in labels:
                # [x, y, w, h]
                bboxes.append(converter.CoordinateConvert.top_xywh2top_xyxy(label['bboxes']))
                classes.append(int(label['category_id']))

            bboxes = np.array(bboxes, dtype=int)

            yield dict(
                _id=tmp['file_name'],
                image=image,
                size=size,
                bboxes=bboxes,
                classes=classes,
            )
