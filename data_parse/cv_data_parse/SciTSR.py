import os
import json
import numpy as np
from pathlib import Path
from utils import os_lib
from .base import DataRegister, DataLoader, DataSaver


class Loader(DataLoader):
    """https://github.com/Academic-Hammer/SciTSR

    Data structure:
        .
        ├── SciTSR-COMP.list    # complicated samples, 2885 of train and 716 of test
        ├── test        # 3000 items
        │   ├── chunk   # position and text of per cells
        │   ├── img     # extract by pdf
        │   ├── pdf     # original pdf
        │   └── structure   # index for chunks
        └── train       # 12000 items
            ├── chunk
            ├── img
            ├── pdf
            ├── rel     # relations of each cell
            └── structure

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SciTSR import DataRegister, Loader

            loader = Loader('data/SciTSR')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentation = r['segmentation']
            image = ImageVisualize.box(image, segmentation)

    """

    image_suffix = 'png'

    def _call(self, set_type=DataRegister.TRAIN, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                segmentation: a list with shape of (-1, -1, 2)

        """
        root_dir = f'{self.data_dir}/{set_type.value}'
        gen_func = Path(f'{root_dir}/chunk').glob('*.chunk')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.ARRAY, root_dir='', **kwargs) -> dict:
        with open(fp, 'r', encoding='utf8') as f:
            js = json.load(f)

        image_path = os.path.abspath(f'{root_dir}/pdf/{fp.stem}.pdf')
        if image_type == DataRegister.PATH:
            raise ValueError('image_type only apply for Register.ARRAY not Register.PATH')
        elif image_type == DataRegister.ARRAY:
            image = os_lib.loader.pdf2images(image_path, scale_ratio=1)[0]
        else:
            raise ValueError(f'Unknown input {image_type = }')

        segmentation = []
        for chunk in js['chunks']:
            pos = chunk['pos']
            if pos[1] < pos[0]:
                pos[0], pos[1] = pos[1], pos[0]

            # [x1, x2, y1, y2] -> [x1, y1, x2, y2]
            pos[1], pos[2] = pos[2], pos[1]

            # refer to: https://github.com/Academic-Hammer/SciTSR/issues/1
            pos[1], pos[3] = 842 - pos[1], 842 - pos[3]

            segmentation.append(pos)

        segmentation = np.array(segmentation)

        yield dict(
            _id=fp.stem + f'.{self.image_suffix}',
            image=image,
            segmentation=segmentation
        )
