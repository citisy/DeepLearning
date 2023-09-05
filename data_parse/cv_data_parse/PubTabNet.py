import os
import jsonlines
import json
from .base import DataRegister, DataLoader, DataSaver, get_image


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
            from data_parse.cv_data_parse.Icdar import DataRegister, Loader

            loader = Loader('data/pubtabnet')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentation = r['segmentation']
            image = ImageVisualize.box(image, segmentation)

    """
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST, DataRegister.VAL]

    def _call(self, set_type, image_type, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                segmentation: a list with shape of (-1, -1, 2)
                transcription: List[str]
        """
        # {filename, split, imgid, html}
        gen_func = jsonlines.open(f'{self.data_dir}/PubTabNet_2.0.0.{set_type.value}.jsonl', 'r')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, line, set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(f'{self.data_dir}/{set_type.value}/{line["filename"]}')
        image = get_image(image_path, image_type)

        segmentation = [cell['bbox'] for cell in line['html']['cells'] if 'bbox' in cell]
        transcription = [cell['tokens'] for cell in line['html']['cells'] if 'bbox' in cell]

        return dict(
            _id=line['filename'],
            image=image,
            segmentation=segmentation,
            transcription=transcription
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
