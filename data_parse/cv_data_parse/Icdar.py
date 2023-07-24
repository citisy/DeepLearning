import os
import json
import cv2
import numpy as np
from utils import converter
from .base import DataLoader, DataRegister, get_image
from pathlib import Path


class Icdar2015(DataLoader):
    """https://rrc.cvc.uab.es/?ch=4&com=downloads

    Data structure:
        .
        ├── ch4_training_images                     # task 1(Text Localization) and task 4(End to End) train images, 1000 items
        ├── ch4_test_images                         # task 1  and task 4 test images, 500 items
        ├── ch4_training_localization_transcription_gt  # task 1 train labels
        ├── Challenge4_Test_Task1_GT                # task 1 test labels
        ├── ch4_training_word_images_gt             # task 3(Word Recognition) train images and labels
        ├── ch4_test_word_images_gt                 # task 3 test images
        ├── Challenge4_Test_Task3_GT.txt            # task 3 test labels
        ├── ch4_training_vocabularies_per_image     # task 4(End to End) train labels
        ├── ch4_training_vocabulary.txt             # task 4 train set labels
        ├── ch4_test_vocabularies_per_image         # task 4 test labels
        ├── ch4_test_vocabulary.txt                 # task 4 test set labels
        ├── Challenge4_Test_Task4_GT                # equals to Challenge4_Test_Task1_GT
        └── GenericVocabulary.txt                   # total vocabulary

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.Icdar import DataRegister, Icdar2015 as Loader

            loader = Loader('data/ICDAR2015')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentations = r['segmentations']
            transcriptions = r['transcriptions']

            vis_image = np.zeros_like(image) + 255
            vis_image = ImageVisualize.box(vis_image, segmentations)
            vis_image = ImageVisualize.text(vis_image, segmentations, transcriptions)
    """
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    image_suffix = 'jpg'

    def _call(self, set_type, image_type, train_task='1', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:
            train_task(str):
                one of ['1', '3', '4']
                more info in https://rrc.cvc.uab.es/?ch=4&com=downloads

        Returns:
            see also `self.load_task_1` if train_task='1'
            see also `self.load_task_3` if train_task='3'
            see also `self.load_task_4` if train_task='4'
        """

        if train_task == '1':
            return self.load_task_1(set_type, image_type, **kwargs)
        elif train_task == '3':
            return self.load_task_3(set_type, image_type, **kwargs)
        elif train_task == '4':
            return self.load_task_4(set_type, image_type, **kwargs)
        else:
            raise ValueError(f'dont support {train_task = }')

    def load_task_1(self, set_type, image_type, **kwargs):
        """See Also `self._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                segmentations: a np.ndarray with shape of (-1, 4, 2)
                transcription: List[str]
        """
        if set_type == DataRegister.TRAIN:
            image_dir = 'ch4_training_images'
            label_dir = 'ch4_training_localization_transcription_gt'
        else:
            image_dir = 'ch4_test_images'
            label_dir = 'Challenge4_Test_Task1_GT'

        for fp in Path(f'{self.data_dir}/{label_dir}').glob('*.txt'):
            image_path = os.path.abspath(f'{self.data_dir}/{image_dir}/{fp.stem.replace("gt_", "")}.{self.image_suffix}')
            image = get_image(image_path, image_type)

            with open(fp, 'r', encoding='utf8') as f:
                lines = f.read().strip().strip('\ufeff').split('\n')

            labels = [_.split(',', 8) for _ in lines]
            labels = np.array(labels)

            segmentations = np.array(labels[:, :8], dtype=int).reshape((-1, 4, 2))
            transcriptions = labels[:, 8].tolist()

            yield dict(
                _id=Path(image_path).name,
                image=image,
                segmentations=segmentations,
                transcriptions=transcriptions,
            )

    def load_task_3(self, set_type, image_type, **kwargs):
        pass

    def load_task_4(self, set_type, image_type, **kwargs):
        pass


Loader = Icdar2015
