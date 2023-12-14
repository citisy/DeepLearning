import gzip
import struct
import numpy as np
from .base import DataRegister, DataLoader, DataSaver

info = {
    # http://yann.lecun.com/exdb/mnist/
    'mnist': [
        {
            'url': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'fp': 'train-images-idx3-ubyte.gz',
            'md5': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
            'len': 60000,
            'classes': list(range(10))

        },

        {
            'url': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'fp': 'train-labels-idx1-ubyte.gz',
            'md5': 'd53e105ee54ea40749a09fcbcd1e9432',
            'len': 60000,
            'classes': list(range(10))
        },

        {
            'url': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'fp': 't10k-images-idx3-ubyte.gz',
            'md5': '9fb629c4189551a2d022fa330f9573f3',
            'len': 10000,
            'classes': list(range(10))
        },

        {
            'url': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            'fp': 't10k-labels-idx1-ubyte.gz',
            'md5': 'ec29112dd5afa0611ce80d1b7f02629c',
            'len': 10000,
            'classes': list(range(10))
        }
    ],

    # https://github.com/zalandoresearch/fashion-mnist/
    'fashion-mnist': [
        {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'fp': 'train-images-idx3-ubyte.gz',
            'md5': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
            'len': 60000,
            'classes': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        },

        {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'fp': 'train-labels-idx1-ubyte.gz',
            'md5': '25c81989df183df01b3e8a0aad5dffbe',
            'len': 60000,
            'classes': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        },

        {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'fp': 't10k-images-idx3-ubyte.gz',
            'md5': 'bef4ecab320f06d8554ea6380940ec79',
            'len': 10000,
            'classes': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        },

        {
            'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
            'fp': 't10k-labels-idx1-ubyte.gz',
            'md5': 'bb300cfdad3c16e7a12a480ee83cd310',
            'len': 10000,
            'classes': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        }
    ],

    # https://github.com/rois-codh/kmnist
    'kmnist': [
        {
            'url': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
            'fp': 'train-images-idx3-ubyte.gz',
            'md5': 'bdb82020997e1d708af4cf47b453dcf7',
            'len': 60000,
            'classes': ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
        },

        {
            'url': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
            'fp': 'train-labels-idx1-ubyte.gz',
            'md5': 'e144d726b3acfaa3e44228e80efcd344',
            'len': 60000,
            'classes': ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
        },

        {
            'url': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
            'fp': 't10k-images-idx3-ubyte.gz',
            'md5': '5c965bf0a639b31b8f53240b1b52f4d7',
            'len': 10000,
            'classes': ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
        },

        {
            'url': 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz',
            'fp': 't10k-labels-idx1-ubyte.gz',
            'md5': '7320c461ea6c1c855c0b718fb2a4b134',
            'len': 10000,
            'classes': ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
        }
    ],

}


class Loader(DataLoader):
    """http://yann.lecun.com/exdb/mnist/

    Data structure:
        .
        ├── t10k-images-idx3-ubyte.gz   # test images, included 10,000 images
        ├── t10k-labels-idx1-ubyte.gz   # test labels, included 10,000 labels
        ├── train-images-idx3-ubyte.gz  # train images, included 60,000 images
        └── train-labels-idx1-ubyte.gz  # train labels, included 60,000 labels

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Mnist import DataRegister, Loader

            loader = Loader('data/mnist')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            image = r['image']
            _class = r['_class']

    """
    default_image_type = DataRegister.ARRAY
    classes = list(range(10))
    dataset_info = info['mnist']

    def _call(self, set_type, image_type, decompression=False, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type: only support DataRegister.ARRAY
            decompression(bool): if true, the data file would like *.gz

        Returns:
            a dict had keys of
                image: see also image_type
                size: image shape
                _class: int
        """

        assert image_type == DataRegister.ARRAY, f"Only support image_type = DataRegister.ARRAY"

        if set_type == DataRegister.TRAIN:
            image_fp = f'{self.data_dir}/train-images-idx3-ubyte'
            label_fp = f'{self.data_dir}/train-labels-idx1-ubyte'
        elif set_type == DataRegister.TEST:
            image_fp = f'{self.data_dir}/t10k-images-idx3-ubyte'
            label_fp = f'{self.data_dir}/t10k-labels-idx1-ubyte'
        else:
            raise ValueError(f'Dont support {set_type = }')

        if decompression:
            image_f = gzip.open(image_fp + '.gz', 'rb')
            label_f = gzip.open(label_fp + '.gz', 'rb')
        else:
            image_f = open(image_fp, 'rb')
            label_f = open(label_fp, 'rb')

        _, num, rows, cols = struct.unpack('>IIII', image_f.read(16))
        images = np.frombuffer(image_f.read(), dtype=np.uint8).reshape(num, 784)
        image_f.close()

        _, n = struct.unpack('>II', label_f.read(8))
        labels = np.frombuffer(label_f.read(), dtype=np.uint8)
        label_f.close()

        for i, (x, y) in enumerate(zip(images, labels)):
            x = x.reshape(28, 28)
            x = np.expand_dims(x, axis=-1)
            yield dict(
                _id=f'{i}.{self.image_suffix}',
                image=x,
                size=x.shape,
                _class=y,
            )
