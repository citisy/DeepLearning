import os
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from .base import DataLoader, DataRegister, get_image


class Loader(DataLoader):
    """https://cmp.felk.cvut.cz/~tylecr1/facade/

    Data structure:
        .
        ├── base            # as train data, 378 items
        │   ├── xxx.jpg		# original image
        │   ├── xxx.png     # pixels image with labels
        │   ├── xxx.xml	    # labels
        │   └── ...
        └── extended        # as test data, 228 items

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.cmp_facade import DataRegister, Loader, DataVisualizer

            data_dir = 'data/cmp_facade'
            loader = Loader(data_dir)
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """

    classes = ['facade', 'molding', 'cornice', 'pillar', 'window', 'door', 'sill', 'blind', 'balcony', 'shop', 'deco', 'background']

    set_type_dict = {
        DataRegister.TRAIN: 'base',
        DataRegister.TEST: 'extended'
    }

    def _call(self, set_type=DataRegister.TRAIN, **kwargs):
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
        gen_func = Path(f'{self.data_dir}/{self.set_type_dict[set_type]}').glob('*.jpg')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        pix_image_path = image_path.replace('.jpg', '.png')
        pix_image = get_image(pix_image_path, image_type)

        label_path = image_path.replace('.jpg', '.xml')
        # get wrong directly with using raw xml file
        with open(label_path, 'r', encoding='utf8') as f:
            xml = f.read()
        tree = ET.fromstring(f'<root>\n{xml}\n</root>')

        bboxes = []
        classes = []

        for obj in tree.iter('object'):
            points = obj.find('points')
            xs = [float(x.text.strip()) for x in points.iter('x')]
            ys = [float(y.text.strip()) for y in points.iter('y')]
            bboxes.append([ys[0], xs[0], ys[1], xs[1]])
            classes.append(int(obj.find('label').text.strip()))

        bboxes = np.array(bboxes)
        classes = np.array(classes) - 1

        return dict(
            _id=fp.name,
            image=image,
            pix_image=pix_image,
            bboxes=bboxes,
            classes=classes,
        )
