import os
import scipy
from pathlib import Path
from .base import DataRegister, DataLoader, DataSaver, get_image


class Loader(DataLoader):
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
            from data_parse.cv_data_parse.ImageNet import DataRegister, ImageNet2012Loader as Loader

            loader = Loader('data/ImageNet2012')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            image = r['image']
            _class = r['_class']
    """
    default_set_type = [DataRegister.TRAIN, DataRegister.VAL]
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

    def _call(self, set_type, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:
            wnid(str):
                special wnid from `meta.mat` would be loaded, only used for set_type=DataRegister.TRAIN

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                _class: index of `self.classes`
        """
        if set_type == DataRegister.TRAIN:
            return self.load_train(**kwargs)
        else:
            return self.load_val(**kwargs)

    def load_train(self, wnid=None, **kwargs):
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

        def gen_func():
            for img_fp, _class in zip(img_list, classes):
                image_path = os.path.abspath(img_fp)
                yield image_path, _class

        return self.gen_data(gen_func(), **kwargs)

    def load_val(self, **kwargs):
        with open(f'{self.data_dir}/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', 'r') as f:
            _classes = f.read().strip().split('\n')

        def gen_func():
            for i, _class in enumerate(_classes):
                image_path = os.path.abspath(f'{self.data_dir}/ILSVRC2012_img_val/ILSVRC2012_val_{i + 1:08d}.{self.image_suffix}')
                yield image_path, int(_class) - 1

        return self.gen_data(gen_func(), **kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path, _class = obj
        image = get_image(image_path, image_type)

        return dict(
            _id=Path(image_path).name,
            image=image,
            _class=_class
        )


class Saver(DataSaver):
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
