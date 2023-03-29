import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib, converter
from cv_data_parse.base import DataRegister, DataLoader, DataSaver
from pathlib import Path
from tqdm import tqdm
import fitz


class Loader(DataLoader):
    """load image, context and the bbox of it from the pdf files
    Data structure:
        .
        └── pdfs
            └── [task]

    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.pdf import DataRegister, Loader

            loader = Loader('data/pdf')
            data = loader(set_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
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
    default_set_type = [DataRegister.place_holder]
    image_suffix = 'png'
    pdf_suffix = 'pdf'

    def _call(self, *args, task='', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            task(str): one of dir name in `pdfs` dir

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                segmentations: a np.ndarray with shape of (-1, 4, 2)
                segmentations_: List[np.ndarray] of chars
                transcriptions: List[str]
        """
        for fp in Path(f'{self.data_dir}/pdfs/{task}').glob(f'*.{self.pdf_suffix}'):
            images = os_lib.PdfOS.pdf2images2(str(fp))
            doc = fitz.open(str(fp))

            for i, (image, page) in enumerate(zip(images, doc)):
                data_dic = dict(
                    _id=f'{fp.stem}_{i}.png',
                    image=image,
                )
                data_dic.update(self.load_per_page(page))

                yield data_dic

    def load_per_page(
            self, source: fitz.fitz.Page or dict,
            scale_ratio: float = 1.33333333,
            shrink: bool = True,
    ):
        transcriptions = []
        segmentations_ = []
        segmentations = []

        if isinstance(source, fitz.fitz.Page):
            content = source.get_text('rawdict')  # content(dict): 'width', 'height', 'blocks'
        elif isinstance(source, dict):
            content = source
        else:
            raise ValueError('content type error, please check about it')

        for block in content['blocks']:  # block(dict): 'number', 'type', 'bbox', 'lines'
            if block['type'] != 0:  # not a text block
                continue
            for line in block['lines']:  # line(dict): 'spans', 'wmode', 'dir', 'bbox'
                char_box = []
                text = []

                for span in line['spans']:  # span(dict): 'size', 'flags', 'font', 'color',
                    # 'ascender', 'descender', 'chars', 'origin', 'bbox'
                    ascender = span['ascender']
                    descender = span['descender']
                    size = span['size']

                    start = False
                    for char in span['chars']:  # char(dict): 'origin', 'bbox', 'c'
                        if char['c'] == ' ' and not start:
                            continue

                        start = True

                        text.append(char['c'])
                        if not shrink:
                            char_box.append(list(char['bbox']))
                        else:
                            x0, y0, x1, y1 = char['bbox']
                            y_origin = char['origin'][1]
                            y0, y1 = self.shrink_bbox(ascender, descender, size, y0, y1, y_origin)
                            char_box.append((x0, y0, x1, y1))

                if text:
                    transcriptions.append(text)
                    segmentations_.append(char_box)
                    segmentations.append(list(line['bbox']))

        segmentations_ = [np.array(i) * scale_ratio for i in segmentations_]
        segmentations = np.array(segmentations) * scale_ratio
        segmentations = converter.CoordinateConvert.rect2box(segmentations)
        segmentations = segmentations.astype(int)
        transcriptions = [''.join(i) for i in transcriptions]

        if segmentations.size == 0:
            segmentations = np.zeros((0, 4))

        return dict(
            transcriptions=transcriptions,
            segmentations=segmentations,
            segmentations_=segmentations_
        )

    @staticmethod
    def shrink_bbox(
            ascender: float, descender: float, size: float,
            y0: float, y1: float, y_origin: float
    ) -> tuple:
        # shrink bbox to the reduced glyph heights
        # details on https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
        if size >= y1 - y0:  # don't need to shrink
            return y0, y1
        else:
            new_y1 = y_origin - size * descender / (ascender - descender)
            new_y0 = new_y1 - size
            return new_y0, new_y1

