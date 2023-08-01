import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .base import DataRegister, DataLoader, DataSaver, get_image, DataVisualizer

DET = 1
SEG_CLS = 2
SEG_OBJ = 3


class Loader(DataLoader):
    """http://host.robots.ox.ac.uk/pascal/VOC/

    Data structure(bass on VOC2012):
        .
        ├── Annotations               # xml files, included bboxeses and lables
        ├── ImageSets                 # subclass sets
        │   ├── Action                # human actions sets
        │   ├── Layout                # human layout sets
        │   ├── Main                  # object detection sets
        │   │     ├── *train.txt      # 5717 items, the first column is file stem, the second column gives whether contained the object or not, -1 gives not contained
        │   │     ├── *val.txt        # 5823 items
        │   │     └── *trainval.txt   # 11540 items
        │   └── Segmentation          # segmentation sets
        │         ├── train.txt       # 1464 items, per image file stem per line
        │         ├── val.txt         # 1449 items
        │         └── trainval.txt    # 2913 items
        ├── JPEGImages                # original images, 17125 items
        ├── SegmentationClass         # images after segmentation base on class
        └── SegmentationObject        # images after segmentation base on object

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Voc import DataRegister, Loader

            loader = Loader('data/VOC2012')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer('data/VOC2012/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.TRAIN_VAL]

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    image_suffix = 'jpg'

    def _call(self, set_type, image_type, set_task=DET, task=None, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:

            set_task(int): 1,2,3
                1, data from JPEGImages for object detection task
                2, data from JPEGImages and SegmentationClass for image segmentation task
                3, data from JPEGImages and SegmentationObject for image segmentation task

            task(None or str): task to output
                None, use Annotations
                str, task from ImageSets.Main, see also `self.classes`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                size: image shape
                bboxes: a np.ndarray with shape of (-1, 4), 4 means [top_left_x, top_left_y, w, h]
                classes: list
                difficult: bool
        """
        if set_task == DET:
            if task is None:
                return self.load_det_total(image_type, **kwargs)
            else:
                return self.load_det_task(set_type, image_type, task, **kwargs)
        else:
            if set_task == SEG_CLS:
                return self.load_seg_task(set_type, image_type, task='SegmentationClass')
            elif set_task == SEG_OBJ:
                return self.load_seg_task(set_type, image_type, task='SegmentationObject')

    def load_det_total(self, image_type, **kwargs):
        for xml_file in Path(f'{self.data_dir}/Annotations').glob('*.xml'):
            ret = self.parse_xml(xml_file.stem, image_type)
            ret = self.convert_func(ret)
            if self.filter_func(ret):
                yield ret

    def load_det_task(self, set_type, image_type, task='', **kwargs):
        if task:
            task += '_'

        with open(f'{self.data_dir}/ImageSets/Main/{task}{set_type.value}.txt', 'r', encoding='utf8') as f:
            for _id in f.read().strip().split('\n'):
                ret = self.parse_xml(_id, image_type)
                ret = self.convert_func(ret)
                if self.filter_func(ret):
                    yield ret

    def load_seg_task(self, set_type, image_type, task='SegmentationClass', **kwargs):
        with open(f'{self.data_dir}/ImageSets/Segmentation/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for _id in f.read().strip().split('\n'):
                ret = self.parse_xml(_id, image_type)

                pix_image_path = os.path.abspath(f'{self.data_dir}/{task}/{_id}.png')
                pix_image = get_image(pix_image_path, image_type)

                ret['pix_image'] = pix_image

                ret = self.convert_func(ret)
                if self.filter_func(ret):
                    yield ret

    def parse_xml(self, _id, image_type):
        xml_file = Path(f'{self.data_dir}/Annotations/{_id}.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_path = os.path.abspath(f'{self.data_dir}/JPEGImages/{_id}.{self.image_suffix}')
        image = get_image(image_path, image_type)

        elem = root.find('size')
        size = {subelem.tag: int(subelem.text) for subelem in elem}
        # h, w, c
        size = (size['width'], size['height'], size['depth'])

        bboxes = []
        classes = []
        difficult = []
        for obj in root.iter('object'):
            obj_name = obj.find('name').text
            difficult.append(int(obj.find('difficult').text) if obj.find('difficult') else 0)
            classes.append(self.classes.index(obj_name))
            xmlbox = obj.find('bndbox')
            bboxes.append([float(xmlbox.find(value).text) for value in ('xmin', 'ymin', 'xmax', 'ymax')])
        bboxes = np.array(bboxes)
        classes = np.array(classes)

        return dict(
            _id=f'{_id}.{self.image_suffix}',
            image=image,
            size=size,
            bboxes=bboxes,
            classes=classes,
            difficult=difficult
        )
