import lmdb
import cv2
import numpy as np
from .base import DataLoader, DataRegister, get_image


class Loader(DataLoader):
    """http://dl.yf.io/lsun/

    Data structure:
        .
        ├── [task]_train_lmdb   # train data, can refer to http://dl.yf.io/lsun/categories.txt to get the task
        │   ├── data.lmdb
        │   └── lock.lmdb
        ├── [task]_val_lmdb     # val data
        │   ├── data.lmdb
        │   └── lock.lmdb
        └── ...

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.lsun import DataRegister, Loader, DataVisualizer

            data_dir = 'data/lsun'
            loader = Loader(data_dir)
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    classes = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']
    default_set_type = [DataRegister.TRAIN, DataRegister.VAL]

    def _call(self, set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, task='bedroom', **kwargs):
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

        db_path = f'{self.data_dir}/{task}_{set_type.value}_lmdb'
        env = lmdb.open(db_path, map_size=1099511627776, max_readers=100, readonly=True)

        txn = env.begin(write=False)
        gen_func = txn.cursor()
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        key, val = obj
        image = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)

        return dict(
            _id=f'{key.decode("utf8")}.{self.image_suffix}',
            image=image,
        )
