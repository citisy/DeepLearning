import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib, converter
from .base import DataRegister, DataLoader, DataSaver
from tqdm import tqdm
from pathlib import Path


class Loader(DataLoader):
    """https://github.com/PaddleNLP/PaddleNLP

    Data structure:
        .
        ├── images
        ├── label_studio.json
        └── labels
            └── [set_task]
                  ├── train.txt  # per image file path per line
                  ├── test.txt   # would like to be empty or same to val.txt
                  └── dev.txt

    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]
    image_suffix = 'png'

    def _call(self, set_type, image_type=DataRegister.BASE64, set_task='', **kwargs):
        """See Also `data_parse.cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            set_task(str): one of dir name in `labels` dir
            load_method: default `self.load_det`

        Returns:
            return of load_method

        """
        with open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for i, s in enumerate(f.readlines()):
                js = json.loads(s)

                image = js['image']
                if image_type == DataRegister.ARRAY:
                    image = converter.DataConvert.base64_to_image(image)
                elif image_type == DataRegister.BASE64:
                    pass
                else:
                    raise ValueError(f'Unknown input {image_type = }')

                bboxes = js['bbox']
                text = js['content']
                result_list = js['result_list']
                prompt = js['prompt']
                bboxes = np.array(bboxes)
                # bboxes *= np.array([4, 3, 4, 3])

                yield dict(
                    _id=f'{i}.{self.image_suffix}',
                    image=image,
                    bboxes=bboxes,
                    text=text,
                    result_list=result_list,
                    prompt=prompt
                )


class Saver(DataSaver):
    def _call(self, iter_data, set_type, image_type, task='', set_task='', **kwargs):
        pass
