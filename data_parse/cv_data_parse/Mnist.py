import gzip
import struct
import numpy as np
from .base import DataRegister, DataLoader, DataSaver


class Loader(DataLoader):
    """http://yann.lecun.com/exdb/mnist/
    See Also https://github.com/zalandoresearch/fashion-mnist/

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
            data = loader(set_type=DataRegister.ALL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            image = r['image']
            _class = r['_class']

    """
    default_image_type = DataRegister.ARRAY
    classes = list(range(10))

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
