import fitz
import yaml
import cv2
import base64
import pickle
import uuid
import json
import os
import configparser
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes
from collections import abc
from configparser import ExtendedInterpolation


def mk_dir(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


class IniConfigParser(configparser.ConfigParser):
    def __init__(self):
        super(IniConfigParser, self).__init__(interpolation=ExtendedInterpolation())

    def load_config(self, config_path, return_dict=True):
        assert Path(config_path).is_file(), f"Config file is missing: {config_path}"
        self.read(config_path)
        if return_dict:
            return self._sections
        else:
            return self


class YmlConfigParser:
    def __init__(self):
        self.config = {}

    def load_config(self, config_path):
        config = (yaml.load(open(config_path, 'rb'), Loader=yaml.Loader))
        self.config.update(config)
        return self.config


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
                   rotate: int = 0, alpha=False) -> np.ndarray:
        trans = fitz.Matrix(scale_ratio, scale_ratio).prerotate(rotate)  # rotate means clockwise
        pm = page.get_pixmap(matrix=trans, alpha=alpha)
        data = pm.tobytes()  # bytes
        img_np = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_np, flags=-1)
        return img

    @staticmethod
    def page2bgr_image(page: fitz.fitz.Page, scale_ratio: float = 1.33333333,
                       rotate: int = 0, alpha=True) -> np.ndarray:
        trans = fitz.Matrix(scale_ratio, scale_ratio).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=alpha)
        img = pm.getPNGData()
        img_np = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img_np, flags=1)
        return img

    @classmethod
    def page2image4lined_table(cls, page: fitz.fitz.Page, scale_ratio: float = 1.33333333) -> np.ndarray:
        def bgra2bgr(img):
            alpha = img[:, :, 3]
            if True:
                _, alpha = cv2.threshold(alpha, 50, 255, cv2.THRESH_BINARY)
            alpha = alpha / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            source = img[:, :, :3] / 255.0
            bg = np.array([255, 255, 255]) / 255.0
            target = (1 - alpha) * bg + alpha * source
            ret = (target * 255).astype(np.uint8)
            return ret

        img = cls.page2image(page, scale_ratio=scale_ratio, rotate=0, alpha=True)
        if img.shape[2] == 4:
            img = bgra2bgr(img)
            return img
        else:
            return img

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
        """use pdf2image to convert pdf to images"""
        dpi = 72 * scale_ratio
        if isinstance(src, str):
            images = convert_from_path(src, dpi=dpi)
        else:
            images = convert_from_bytes(src, dpi=dpi)

        return [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]

    @classmethod
    def pdf2bgr_images(cls, pdf_path,
                       scale_ratio: float = 1.33333333,
                       rotate: int = 0
                       ):
        """pdf转图片,图片格式BGR"""
        doc = fitz.open(pdf_path)
        return [cls.page2image(page, scale_ratio=scale_ratio, rotate=rotate) for page in tqdm(doc, desc="pdf -> images")]

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


cache_json = Cache.cache_json


class ImageListDisk(abc.MutableSequence):
    """A list-like object.
    Context: I store images extracted from pdf file. If a pdf contains too many
    pages, it can cause OOM. My code deals with images by using list, so I want
    to keep it that way, but under certain situations, the 'list' actually stores
    images on disk but not in memory, what it actually stores are the save paths
    of images.
    """

    def __init__(self, imgs, save_dir='CACHE'):
        mk_dir(save_dir)
        self.save_dir = save_dir
        self.save_paths = [self._save_object(i) for i in imgs]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            new_l = ImageListDisk([], self.save_dir)
            new_l.save_paths = self.save_paths[idx]
            return new_l
        save_path = self.save_paths[idx]
        o = self._load_object(save_path)
        return o

    def __len__(self):
        return self.save_paths.__len__()

    def __setitem__(self, idx, o):
        if isinstance(idx, slice) and isinstance(o, ImageListDisk):
            self.save_paths[idx] = o.save_paths
        elif isinstance(idx, slice) and not isinstance(o, ImageListDisk):
            save_paths = [self._save_object(i) for i in o]
            self.save_paths[idx] = save_paths
        else:
            self.save_paths[idx] = self._save_object(o)

    def __delitem__(self, idx):
        del self.save_paths[idx]

    def insert(self, idx, o):
        save_path = self._save_object(o)
        self.save_paths.insert(idx, save_path)

    def __repr__(self):
        return self.save_paths.__repr__()

    def _save_object(self, o):
        save_name = str(uuid.uuid1()) + '.pickle'
        save_path = Path(self.save_dir) / save_name
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(o, pickle_file)
        return save_path

    def _load_object(self, save_path):
        with open(save_path, 'rb') as pickle_file:
            o = pickle.load(pickle_file)
        return o

    def close(self):
        for fp in self.save_paths:
            if fp.exists():
                fp.unlink()


class EmptyOs:
    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
