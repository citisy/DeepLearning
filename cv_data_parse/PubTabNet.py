import os
import jsonlines
import json
import cv2
from cv_data_parse.base import DataRegister, DataLoader, DataSaver


class Loader(DataLoader):
    """https://github.com/ibm-aur-nlp/PubTabNet

    Data structure:
        .
        ├── PubTabNet_2.0.0.jsonl
        ├── test    # 9138 items, no labels information in json file
        ├── train   # 500777 items
        └── val     # 9115 items

    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.Icdar import DataRegister, Loader

            loader = Loader('data/pubtabnet')
            data = loader(data_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentation = r['segmentation']
            image = ImageVisualize.box(image, segmentation)

    """
    default_load_type = [DataRegister.TRAIN, DataRegister.TEST, DataRegister.VAL]

    def _call(self, load_type, image_type, **kwargs):
        with jsonlines.open(f'{self.data_dir}/PubTabNet_2.0.0.{load_type.value}.jsonl', 'r') as reader:
            # {filename, split, imgid, html}
            for line in reader:

                image_path = os.path.abspath(f'{self.data_dir}/{load_type.value}/{line["filename"]}')
                if image_type == DataRegister.PATH:
                    image = image_path
                elif image_type == DataRegister.IMAGE:
                    image = cv2.imread(image_path)
                else:
                    raise ValueError(f'Unknown input {image_type = }')

                segmentation = [cell['bbox'] for cell in line['html']['cells'] if 'bbox' in cell]
                text = [cell['tokens'] for cell in line['html']['cells'] if 'bbox' in cell]

                yield dict(
                    _id=line['filename'],
                    image=image,
                    segmentation=segmentation,
                    text=text
                )

    def split_json(self):
        f1 = open(f'{self.data_dir}/PubTabNet_2.0.0.train.jsonl', 'w', encoding='utf8')
        f2 = open(f'{self.data_dir}/PubTabNet_2.0.0.test.jsonl', 'w', encoding='utf8')
        f3 = open(f'{self.data_dir}/PubTabNet_2.0.0.val.jsonl', 'w', encoding='utf8')

        with open(f'{self.data_dir}/PubTabNet_2.0.0.jsonl', 'r', encoding='utf8') as f:
            line = f.readline()
            while line:
                js = json.loads(line)

                if js['split'] == 'train':
                    f1.write(line)
                elif js['split'] == 'test':  # no data
                    f2.write(line)
                elif js['split'] == 'val':
                    f3.write(line)
                else:
                    raise Exception(line)

                line = f.readline()
