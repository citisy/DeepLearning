import os
import cv2
import pandas as pd
from pathlib import Path
from utils import converter
from .base import DataLoader, DataRegister


class CelebALoader(DataLoader):
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
            ├── img_align_celeba            # align images
            ├── img_align_celeba_png        # cropped images
            └── img_celeba                  # original images

    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.CelebA import DataRegister, CelebALoader as Loader

            loader = Loader('data/CelebA')
            data = loader(data_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            image = r['image']
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

    def _call(self, load_type, image_type, img_task='original', **kwargs):
        """

        Args:
            load_type: no useful, you can set to None
            image_type: DataRegister.FILE or DataRegister.IMAGE
            img_task: which image dir to load, see also `CelebALoader.img_task_dict`

        """
        with open(f'{self.data_dir}/Eval/list_eval_partition.txt') as f:
            lines = f.read().strip().split('\n')

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

        for line in lines:
            _id, _data_type = line.split(' ')

            if img_task == 'crop':
                image_path = os.path.abspath(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}/{Path(_id).stem}.{self.image_suffix}')
            else:
                image_path = os.path.abspath(f'{self.data_dir}/Img/{self.img_task_dict[img_task]}/{_id}')

            if image_type == DataRegister.PATH:
                image = image_path
            elif image_type == DataRegister.IMAGE:
                image = cv2.imread(image_path)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            xywh = list(bbox_df.loc[_id])
            bbox = converter.top_xywh2top_xyxy(xywh)
            attr = list(attr_df.loc[_id])
            identity = identity_df.loc[_id][1]
            landmarks = list(landmarks_df.loc[_id])
            _type = load_type_df.loc[_id][1]

            yield dict(
                _id=_id,
                _type=_type,
                image=image,
                bbox=bbox,
                attr=attr,
                identity=identity,
                landmarks=landmarks
            )


class CelebAHQLoader(DataLoader):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
    """


class CelebASpoofLoader(DataLoader):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html
    """


Loader = CelebALoader
