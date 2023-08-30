import pickle
import numpy as np
from .base import DataLoader, DataRegister
from .data_augmentation.channel import CHW2HWC

chw2hwc = CHW2HWC()


class Cifar10Loader(DataLoader):
    """http://www.cs.toronto.edu/~kriz/cifar.html

    Data structure:
        .
        ├── batches.meta
        ├── data_batch_1    # train images, 10000 items per batch, totally 5 batches
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        └── test_batch      # test images, 10000 items

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Cifar import DataRegister, Cifar10Loader as Loader

            loader = Loader('data/cifar-10-batches-py')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            image = r['image']
            _class = r['_class']
    """
    default_image_type = DataRegister.ARRAY
    image_suffix = 'png'
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def _call(self, set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                size: image shape
                _class: index of `self.classes`
        """

        assert image_type == DataRegister.ARRAY, f"Only support image_type = DataRegister.ARRAY"

        if set_type == DataRegister.TRAIN:
            data_dict = dict()
            for i in range(1, 6):
                with open(f'{self.data_dir}/data_batch_{i}', 'rb') as fo:
                    if data_dict:
                        tmp = pickle.load(fo, encoding='bytes')
                        for k, v in data_dict.items():
                            if k == b'batch_label':
                                continue
                            elif k == b'data':
                                data_dict[k] = np.r_[v, tmp[k]]
                            else:
                                v.extend(tmp[k])
                    else:
                        data_dict = pickle.load(fo, encoding='bytes')

        elif set_type == DataRegister.TEST:
            with open(f'{self.data_dir}/test_batch', 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')

        gen_func = zip(data_dict[b'data'], data_dict[b'labels'], data_dict[b'filenames'])
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        image, _class, _id = obj
        _id = _id.decode('utf8')

        image = np.reshape(image, [3, 32, 32])

        # (c, h, w) -> (h, w, c)
        image = chw2hwc.apply_image(image)

        return dict(
            _id=_id,
            image=image,
            size=image.shape,
            _class=_class,
        )


class Cifar100Loader(Cifar10Loader):
    pass


Loader = Cifar10Loader
