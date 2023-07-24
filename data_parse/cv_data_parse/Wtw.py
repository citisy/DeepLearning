import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
from .base import DataRegister, DataLoader, DataSaver, get_image


class Loader(DataLoader):
    """https://tianchi.aliyun.com/dataset/108587

    Data structure:
        .
        ├── test
        │   ├── images
        │   └── xml     # 3611 items
        └── train
            ├── images
            └── xml     # 10970 items

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Wtw import DataRegister, Loader

            loader = Loader('data/WTW')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentation = r['segmentation']
            transcription = r['transcription']

            vis_image = np.zeros_like(image) + 255
            vis_image = ImageVisualize.box(vis_image, segmentation)
            vis_image = ImageVisualize.text(vis_image, segmentation, transcription)

    """

    def _call(self, set_type, image_type, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                size: image shape
                segmentation: a list with shape of (-1, 4)
        """

        root_dir = f'{self.data_dir}/{set_type.value}'
        for xml_file in Path(f'{root_dir}/xml').glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            assert root.tag == 'annotation', f'pascal voc xml root element should be annotation, rather than {root.tag = }'

            elem = root.find('filename')
            image_path = os.path.abspath(f'{root_dir}/images/{elem.text}')
            image = get_image(image_path, image_type)

            _id = elem.text

            elem = root.find('size')
            size = {subelem.tag: int(subelem.text) for subelem in elem}
            # width, height, depth
            size = (size['width'], size['height'], size['depth'])

            segmentation = []
            for elem in root.iterfind('object'):
                subelem = elem.find('bndbox')
                segmentation.append([float(subelem.find(value).text) for value in ('xmin', 'ymin', 'xmax', 'ymax')])

            yield dict(
                _id=_id,
                image=image,
                size=size,
                segmentation=segmentation
            )
