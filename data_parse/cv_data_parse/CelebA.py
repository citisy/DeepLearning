import os
import cv2
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from utils import cv_utils, converter
from .base import DataLoader, DataRegister, get_image
import io


class Loader(DataLoader):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Data structure:
        .
        ├── Anno
        │   ├── identity_CelebA.txt         # person name class
        │   ├── list_attr_celeba.txt        # attribution class
        │   ├── list_bbox_celeba.txt        # bbox for original images
        │   ├── list_landmarks_align_celeba.txt     # landmarks(eye, nose, mouth) of align images
        │   └── list_landmarks_celeba.txt           # landmarks(eye, nose, mouth) of original images
        ├── Eval
        │   └── list_eval_partition.txt     # dataset partition, 202599 items
        └── Img
            ├── img_align_celeba            # align images, 202602 items
            ├── img_align_celeba_png        # cropped images, 202602 items
            └── img_celeba                  # original images, 202602 items

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.CelebA import DataRegister, CelebALoader as Loader, DataVisualizer
            from data_parse.cv_data_parse.base import DataVisualizer

            loader = Loader('data/CelebA')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)

            # visual
            DataVisualizer('data/CelebAData/visuals', verbose=False, pbar=False)(data[0])
    """
    default_data_type = DataRegister.MIX

    img_task_dict = {
        'original': 'img_celeba',
        'align': 'img_align_celeba',
        'crop': 'img_align_celeba_png'
    }

    image_suffix = 'png'

    attr_classes = None
    landmarks_classes = None

    def _call(self, img_task='original', only_image=False, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            img_task(str): which image dir to load
                see also `CelebALoader.img_task_dict`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                _type
                bbox
                attr
                identity
                landmarks
        """
        with open(f'{self.data_dir}/Eval/list_eval_partition.txt') as f:
            lines = f.read().strip().split('\n')

        if only_image:
            bbox_df, attr_df, identity_df, load_type_df, landmarks_df = [None] * 5
        else:
            bbox_df = pd.read_csv(f'{self.data_dir}/Anno/list_bbox_celeba.txt', sep=r'\s+', index_col=0, header=1)

            attr_df = pd.read_csv(f'{self.data_dir}/Anno/list_attr_celeba.txt', sep=r'\s+', index_col=0, header=1)
            self.attr_classes = tuple(attr_df.columns)

            identity_df = pd.read_csv(f'{self.data_dir}/Anno/identity_CelebA.txt', sep=r'\s+', index_col=0, header=None)
            self.landmarks_classes = tuple(identity_df.columns)

            load_type_df = pd.read_csv(f'{self.data_dir}/Eval/list_eval_partition.txt', sep=r'\s+', index_col=0, header=None)

            if img_task == 'align':
                landmarks_df = pd.read_csv(f'{self.data_dir}/Anno/list_landmarks_align_celeba.txt', sep=r'\s+', index_col=0, header=1)
            else:
                landmarks_df = pd.read_csv(f'{self.data_dir}/Anno/list_landmarks_celeba.txt', sep=r'\s+', index_col=0, header=1)

        gen_func = lines
        return self.gen_data(gen_func, img_task=img_task, only_image=only_image,
                             bbox_df=bbox_df, attr_df=attr_df, identity_df=identity_df,
                             load_type_df=load_type_df, landmarks_df=landmarks_df, **kwargs)

    def get_ret(self, line, image_type=DataRegister.ARRAY, img_task='original', only_image=False,
                bbox_df=None, attr_df=None, identity_df=None, load_type_df=None, landmarks_df=None, **kwargs) -> dict:
        _id, _data_type = line.split(' ')

        if img_task == 'crop':
            image_path = os.path.abspath(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}/{Path(_id).stem}.{self.image_suffix}')
        else:
            image_path = os.path.abspath(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}/{_id}')

        image = get_image(image_path, image_type)

        ret = dict(
            _id=_id,
            image=image,
        )

        if not only_image:
            xywh = list(bbox_df.loc[_id])
            bbox = cv_utils.CoordinateConvert.top_xywh2top_xyxy(xywh)
            attr = list(attr_df.loc[_id])
            identity = identity_df.loc[_id][1]
            landmarks = list(landmarks_df.loc[_id])
            _type = load_type_df.loc[_id][1]

            ret.update(
                _type=_type,
                bbox=bbox,
                attr=attr,
                identity=identity,
                landmarks=landmarks
            )

        return ret


class ZipLoader(Loader):
    """
    Data structure:
        .
        ├── Anno
        │   ├── identity_CelebA.txt         # person name class
        │   ├── list_attr_celeba.txt        # attribution class
        │   ├── list_bbox_celeba.txt        # bbox for original images
        │   ├── list_landmarks_align_celeba.txt     # landmarks(eye, nose, mouth) of align images
        │   └── list_landmarks_celeba.txt           # landmarks(eye, nose, mouth) of original images
        ├── Eval
        │   └── list_eval_partition.txt     # dataset partition, 202599 items
        └── Img
            ├── img_align_celeba.zip           # align images, 202602 items
            ├── img_align_celeba_png.7z        # cropped images, 202602 items
            └── img_celeba.7z                  # original images, 202602 items
    """

    def _call(self, img_task='original', **kwargs):
        if img_task == 'align':
            zip_file = ZipFile(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}.zip', 'r')
        else:
            import py7zr
            zip_file = py7zr.SevenZipFile(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}.7z', mode='r')

        return super()._call(img_task=img_task, zip_file=zip_file, **kwargs)

    def get_ret(self, line, image_type=DataRegister.ARRAY, img_task='original', only_image=False,
                bbox_df=None, attr_df=None, identity_df=None, load_type_df=None, landmarks_df=None,
                zip_file=None, **kwargs) -> dict:
        _id, _data_type = line.split(' ')

        if img_task == 'crop':
            p = f'{self.img_task_dict[img_task]}/{Path(_id).stem}.{self.image_suffix}'
        else:
            p = f'{self.img_task_dict[img_task]}/{_id}'

        if img_task == 'align':
            image = zip_file.open(p).read()
        else:
            # todo: some bugs
            from py7zr.py7zr import MemIO

            for f in zip_file.files:
                if f.filename == p:
                    _buf = io.BytesIO()
                    zip_file.worker.register_filelike(f.id, MemIO(_buf))

                    zip_file.worker.extract(open(zip_file.filename, 'rb'), None, True)
                    image = _buf.read()
                    break
            else:
                raise

        image = converter.DataConvert.bytes_to_image(image)

        ret = dict(
            _id=_id,
            image=image,
        )

        if not only_image:
            xywh = list(bbox_df.loc[_id])
            bbox = cv_utils.CoordinateConvert.top_xywh2top_xyxy(xywh)
            attr = list(attr_df.loc[_id])
            identity = identity_df.loc[_id][1]
            landmarks = list(landmarks_df.loc[_id])
            _type = load_type_df.loc[_id][1]

            ret.update(
                _type=_type,
                bbox=bbox,
                attr=attr,
                identity=identity,
                landmarks=landmarks
            )

        return ret
