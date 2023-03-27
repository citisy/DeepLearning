import os
import cv2
import scipy
import pickle
from pathlib import Path
from utils import os_lib
from cv_data_parse.base import DataRegister, DataLoader, DataSaver


class ImageNet2012Loader(DataLoader):
    """https://www.image-net.org/

    Data structure:
        .
        ├── ILSVRC2012_bbox_test_dogs
        ├── ILSVRC2012_bbox_train_dogs
        ├── ILSVRC2012_bbox_train_v2
        ├── ILSVRC2012_bbox_val_v3
        ├── ILSVRC2012_devkit_t12
        │   └── data
        │       ├── meta.mat                                    # synsets, included 'ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images'
        │       └── ILSVRC2012_validation_ground_truth.txt      # val classes
        ├── ILSVRC2012_devkit_t3
        ├── ILSVRC2012_img_test         # test images, 100000 items, have no labels
        ├── ILSVRC2012_img_train        # train images, 1281167 items, included 1000 dirs of different classes
        ├── ILSVRC2012_img_train_t3     #
        └── ILSVRC2012_img_val          # val images, 50000 items

    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.ImageNet import DataRegister, ImageNet2012Loader as Loader

            loader = Loader('data/ImageNet2012')
            data = loader(data_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            image = r['image']
            _class = r['_class']
    """
    default_load_type = [DataRegister.TRAIN, DataRegister.VAL]
    image_suffix = 'JPEG'

    def __init__(self, data_dir):
        super().__init__(data_dir)

        # ('ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images')
        self.synsets = scipy.io.loadmat(f'{data_dir}/ILSVRC2012_devkit_t12/data/meta.mat')['synsets']
        self.classes = []
        self.wnid = []
        for i in self.synsets:
            self.wnid.append(i[0][1][0])
            self.classes.append(i[0][2][0])

    def _call(self, load_type, image_type, wnid=None, **kwargs):
        if load_type == DataRegister.TRAIN:
            return self.load_train(image_type, wnid)
        else:
            return self.load_val(image_type)

    def load_train(self, image_type, wnid=None):
        img_list = []
        classes = []
        image_root_dir = Path(f'{self.data_dir}/ILSVRC2012_img_train')

        if wnid:
            if isinstance(wnid, str):
                _class = self.wnid.index(wnid)
                for img_fp in (image_root_dir / Path(wnid)).glob(f'*.{self.image_suffix}'):
                    img_list.append(img_fp)
                    classes.append(_class)
            else:
                for _id in wnid:
                    _class = self.wnid.index(_id)
                    for img_fp in (image_root_dir / Path(_id)).glob(f'*.{self.image_suffix}'):
                        img_list.append(img_fp)
                        classes.append(_class)

        else:
            for fp in image_root_dir.glob('*'):
                if fp.is_file():
                    continue

                _class = self.wnid.index(fp.name)
                for img_fp in fp.glob(f'*.{self.image_suffix}'):
                    img_list.append(img_fp)
                    classes.append(_class)

        for img_fp, _class in zip(img_list, classes):
            image_path = os.path.abspath(img_fp)
            if image_type == DataRegister.PATH:
                image = image_path
            elif image_type == DataRegister.IMAGE:
                image = cv2.imread(image_path)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            yield dict(
                _id=img_fp.name,
                image=image,
                _class=_class,
            )

    def load_val(self, image_type):
        with open(f'{self.data_dir}/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
            _classes = f.read().strip().split('\n')

        for i, _class in enumerate(_classes):

            image_path = os.path.abspath(f'{self.data_dir}/ILSVRC2012_img_val/ILSVRC2012_val_{i + 1:08d}.{self.image_suffix}')
            if image_type == DataRegister.PATH:
                image = image_path
            elif image_type == DataRegister.IMAGE:
                image = cv2.imread(image_path)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            yield dict(
                _id=Path(image_path).name,
                image=image,
                _class=int(_class) - 1,
            )


class ImageNet2012Saver(DataSaver):
    """https://www.image-net.org/

    Data structure:
        .
        ├── ILSVRC2012_bbox_test_dogs
        ├── ILSVRC2012_bbox_train_dogs
        ├── ILSVRC2012_bbox_train_v2
        ├── ILSVRC2012_bbox_val_v3
        ├── ILSVRC2012_devkit_t12
        │   └── data
        │       ├── meta.mat                                    # synsets, included 'ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images'
        │       └── ILSVRC2012_validation_ground_truth.txt      # val classes
        ├── ILSVRC2012_devkit_t3
        ├── ILSVRC2012_img_test         # test images, 100000 items, have no labels
        ├── ILSVRC2012_img_train        # train images, 1281167 items, included 1000 dirs of different classes
        ├── ILSVRC2012_img_train_t3     #
        └── ILSVRC2012_img_val          # val images, 50000 items

    """


Loader = ImageNet2012Loader
Saver = ImageNet2012Saver
