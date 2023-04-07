import fitz
import yaml
import cv2
import base64
import pickle
import uuid
import json
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes
from collections import abc


def mk_dir(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


class PngOS:
    @staticmethod
    def imread(path: str, channel_fixed_3=True) -> np.ndarray:
        """读取图片文件，适配中文路径，并转化为3通道的array
        :param path:
        :param channel_fixed_3: 是否转化为3通道
        :return:
        """
        # support chinese path
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if not channel_fixed_3:
            return img
        else:
            if img.shape[2] == 3:
                return img
            elif img.shape[2] == 4:  # bgra
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
            elif img.shape[2] == 1:  # gray
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return img
            else:
                raise ValueError("image has weird channel number: %d" % img.shape[2])

    @staticmethod
    def imwrite(path: str, img: np.ndarray, verbose=True, print_func=None):
        cv2.imencode('.png', img)[1].tofile(path)

        if verbose:
            print_func = print_func or print
            print_func(f'Save to {path} successful!')


class PdfOS:
    @staticmethod
    def page2image(page: fitz.fitz.Page, scale_ratio: float = 1.33333333,
                   rotate: int = 0, alpha=False, bgr=False) -> np.ndarray:
        trans = fitz.Matrix(scale_ratio, scale_ratio).prerotate(rotate)  # rotate means clockwise
        pm = page.get_pixmap(matrix=trans, alpha=alpha)
        data = pm.tobytes()  # bytes
        img_np = np.frombuffer(data, dtype=np.uint8)

        flag = 1 if bgr else -1
        return cv2.imdecode(img_np, flags=flag)

    @staticmethod
    def pdf2base64(pdf_path):
        return str(base64.b64encode(open(pdf_path, 'rb').read()), 'utf-8')

    @classmethod
    def pdf2images(cls, src,
                   scale_ratio: float = 1.33333333,
                   rotate: int = 0):
        if isinstance(src, str):
            doc = fitz.open(src)
        else:
            doc = fitz.open(stream=src, filetype='pdf')

        return [cls.page2image(page, scale_ratio=scale_ratio, rotate=rotate) for page in doc]

    @staticmethod
    def pdf2images2(src, scale_ratio: float = 1.33333333):
        """use pdf2image to convert pdf to images, faster than using fitz"""
        dpi = 72 * scale_ratio
        if isinstance(src, str):
            images = convert_from_path(src, dpi=dpi)
        else:
            images = convert_from_bytes(src, dpi=dpi)

        return [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]

    @classmethod
    def save_pdf_to_img(cls, source_path, page=None, image_dir=None, scale_ratio=1.):
        """pdf转成img保存"""
        imgs = cls.pdf2images2(
            source_path,
            scale_ratio=scale_ratio
        )

        if isinstance(page, int):
            imgs = [imgs[page]]

        elif isinstance(page, list):
            imgs = [imgs[_] for _ in page]

        image_dir = image_dir or source_path.replace('pdfs', 'images').replace(Path(source_path).suffix, '')

        mk_dir(image_dir)

        for i, img in enumerate(imgs):
            PngOS.imwrite(f'{image_dir}/{i}.png', img)

    @staticmethod
    def save_pdf_to_pdf(source_path, page=None, save_path=None):
        """pdf转成pdf保存"""
        from PyPDF2 import PdfFileReader, PdfFileWriter

        pdf_reader = PdfFileReader(source_path)

        pdf_writer = PdfFileWriter()

        if isinstance(page, int):
            pdf_writer.addPage(pdf_reader.getPage(page))

        elif isinstance(page, list):
            for i in page:
                pdf_writer.addPage(pdf_reader.getPage(i))

        suffix = Path(source_path).suffix
        save_path = save_path or source_path.replace(suffix, '_' + suffix)

        with open(save_path, 'wb') as out:
            pdf_writer.write(out)

        print(f'Have save to {save_path}!')


class Cache:
    @classmethod
    def cache_json(cls, js, save_dir, save_name=None, max_size=None, verbose=True, print_func=None, **kwargs):
        if save_name:
            mk_dir(save_dir)
            cls.delete_old_file(save_dir, max_size, verbose, print_func)
            save_path = f'{save_dir}/{save_name}.json'
        else:
            save_path = save_dir

        with open(save_path, 'w', encoding='utf8') as f:
            json.dump(js, f, ensure_ascii=False, indent=4)

        if verbose:
            print_func = print_func or print
            print_func(f'Have save to {save_path}!')

    @staticmethod
    def delete_old_file(save_dir, max_size=None, verbose=True, print_func=None):
        if not max_size:
            return

        caches = [str(_) for _ in Path(save_dir).glob('*.pdf')]

        if len(caches) > max_size:
            ctime = [os.path.getctime(fp) for fp in caches]
            min_ctime = min(ctime)
            old_path = caches[ctime.index(min_ctime)]
            os.remove(old_path)

            if verbose:
                print_func = print_func or print
                print_func(f'Have delete {old_path}!')

            return old_path


class FakeIo:
    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
