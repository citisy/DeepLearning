import cv2
import json
import os
import pickle
import time
import uuid
import yaml
import configparser
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any
from contextlib import nullcontext


def mk_dir(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


def mk_parent_dir(file_path):
    file_path = Path(file_path)
    dir_path = file_path.parent
    mk_dir(dir_path)


suffixes_dict = dict(
    json=('.json', '.js'),
    yml=('.yml', '.yaml'),
    ini=('.ini',),
    txt=('.txt',),
    pkl=('.pkl',),
    joblib=('.joblib',),
    img=('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
    csv=('.csv', '.tsv'),
    xlsx=('.xlsx', '.xls'),
    word=('.docx', '.doc'),
    pdf=('.pdf',),
    np=('.npy', '.npz')
)


def auto_suffix(obj):
    if isinstance(obj, (list, tuple, set)):
        s = suffixes_dict['txt'][0]
    elif isinstance(obj, dict):
        s = suffixes_dict['json'][0]
    elif isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
        s = suffixes_dict['img'][0]
    elif isinstance(obj, pd.DataFrame):
        s = suffixes_dict['csv'][0]
    else:
        s = suffixes_dict['pkl'][0]
    return s


class Saver:
    def __init__(self, verbose=False, stdout_method=print, stdout_fmt='Succeed to save to %s !', stderr_method=print, stderr_fmt='Fail to save to %s !'):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt
        self.stderr_method = stderr_method
        self.stderr_fmt = stderr_fmt

        self.funcs = {
            suffixes_dict['json']: self.save_json,
            suffixes_dict['yml']: self.save_yml,
            suffixes_dict['ini']: self.save_ini,
            suffixes_dict['txt']: self.save_txt,
            suffixes_dict['pkl']: self.save_pkl,
            suffixes_dict['joblib']: self.save_joblib,
            suffixes_dict['img']: self.save_img,
            suffixes_dict['csv']: self.save_csv
        }

    def stdout(self, path):
        self.stdout_method(self.stdout_fmt % path)

    def stderr(self, path):
        self.stderr_method(self.stderr_fmt % path)

    def auto_save(self, obj, path: str, **kwargs):
        suffix = Path(path).suffix.lower()
        mk_parent_dir(path)

        for k, func in self.funcs.items():
            if suffix in k:
                func(obj, path, **kwargs)
                break
        else:
            self.save_bytes(obj, path, **kwargs)

    def save_json(self, obj: dict, path, **kwargs):
        with open(path, 'w', encoding='utf8', errors='ignore') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4, **kwargs)

        self.stdout(path)

    def save_yml(self, obj: dict, path, **kwargs):
        with open(path, 'w') as f:
            yaml.dump(obj, f)
        self.stdout(path)

    def save_ini(self, obj: dict, path, **kwargs):
        parser = configparser.ConfigParser()
        parser.read_dict(obj)

        # 将ConfigParser对象写入文件
        with open(path, 'w', encoding='utf8') as f:
            parser.write(f)

        self.stdout(path)

    def save_txt(self, obj: iter, path, sep='\n', **kwargs):
        with open(path, 'w', encoding='utf8', errors='ignore') as f:
            for o in obj:
                f.write(f'{o}{sep}')

        self.stdout(path)

    def save_pkl(self, obj, path, **kwargs):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)

        self.stdout(path)

    def save_joblib(self, obj, path, **kwargs):
        import joblib
        joblib.dump(obj, path, **kwargs)

        self.stdout(path)

    def save_bytes(self, obj: bytes, path, **kwargs):
        with open(path, 'wb') as f:
            f.write(obj)

        self.stdout(path)

    def save_img(self, obj: np.ndarray, path, **kwargs):
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        cv2.imencode('.png', obj)[1].tofile(path)
        self.stdout(path)
        # flag = cv2.imwrite(path, obj)
        # if flag:
        #     self.stdout(path)
        # else:
        #     self.stderr(path)

    def save_np_array(self, obj: np.ndarray, path, **kwargs):
        np.savetxt(path, obj, **kwargs)
        self.stdout(path)

    def save_csv(self, obj: pd.DataFrame, path, **kwargs):
        obj.to_csv(path, **kwargs)
        self.stdout(path)

    def save_image_from_pdf(self, path, page=None, image_dir=None, scale_ratio=1.33):
        """select pages from pdf, and save with image type"""
        images = loader.load_images_from_pdf2(
            path,
            scale_ratio=scale_ratio
        )

        if isinstance(page, int):
            images = [images[page]]

        elif isinstance(page, list):
            images = [images[_] for _ in page]

        image_dir = image_dir or path.replace('pdfs', 'images').replace(Path(path).suffix, '')
        mk_dir(image_dir)

        for i, img in enumerate(images):
            self.save_img(img, f'{image_dir}/{i}.png')

    def save_pdf_from_pdf(self, path, page=None, save_path=None):
        """select pages from pdf, and save with pdf type"""
        from PyPDF2 import PdfFileReader, PdfFileWriter, errors

        try:
            pdf_reader = PdfFileReader(path)
            pdf_writer = PdfFileWriter()

            if isinstance(page, int):
                pdf_writer.addPage(pdf_reader.getPage(page))

            elif isinstance(page, list):
                for i in page:
                    pdf_writer.addPage(pdf_reader.getPage(i))

            suffix = Path(path).suffix
            save_path = save_path or str(path).replace(suffix, '_' + suffix)

            with open(save_path, 'wb') as out:
                pdf_writer.write(out)

            self.stdout(save_path)
        except errors.PdfReadError:
            self.stderr(save_path)
        except KeyError:
            self.stderr(save_path)

    def save_image_to_pdf(self, obj: bytes or str, path):
        import fitz  # pip install PyMuPDF

        if isinstance(obj, str):
            img_doc = fitz.open(obj)
        else:
            img_doc = fitz.open(stream=obj, filetype='png')

        img_pdf = fitz.open(stream=img_doc.convert_to_pdf(), filetype='pdf')

        with fitz.open() as doc:
            doc.insert_pdf(img_pdf)
            doc.save(path)
        self.stdout(path)

    def save_images_to_pdf(self, obj: List, path):
        import fitz  # pip install PyMuPDF

        doc = fitz.open()
        for img in obj:
            if isinstance(img, str):
                img_doc = fitz.open(img)
            else:
                img_doc = fitz.open(stream=img, filetype='png')

            img_pdf = fitz.open(stream=img_doc.convert_to_pdf(), filetype='pdf')
            doc.insert_pdf(img_pdf)
        doc.save(path)
        self.stdout(path)


class Loader:
    def __init__(self, verbose=False, stdout_method=print, stdout_fmt='Read %s successful!'):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt

        self.funcs = {
            suffixes_dict['json']: self.load_json,
            suffixes_dict['yml']: self.load_yaml,
            suffixes_dict['ini']: self.load_ini,
            suffixes_dict['txt']: self.load_txt,
            suffixes_dict['pkl']: self.load_pkl,
            suffixes_dict['img']: self.load_img,
        }

    def stdout(self, path):
        self.stdout_method(self.stdout_fmt % path)

    def auto_load(self, path: str):
        suffix = Path(path).suffix.lower()

        for k, func in self.funcs.items():
            if suffix in k:
                obj = func(path)
                break
        else:
            obj = self.load_bytes(path)

        return obj

    def load_json(self, path) -> dict:
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            obj = json.load(f)
        self.stdout(path)

        return obj

    def load_yaml(self, path) -> dict:
        obj = yaml.load(open(path, 'rb'), Loader=yaml.Loader)
        self.stdout(path)

        return obj

    def load_ini(self, path) -> configparser.ConfigParser:
        obj = configparser.ConfigParser()
        obj.read(path, encoding="utf-8")
        self.stdout(path)

        return obj

    def load_txt(self, path) -> iter:
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            obj = f.read().rstrip().split('\n')

        self.stdout(path)
        return obj

    def load_txt_dir(self, dirt: str or Path, fmt='*.txt'):
        txts = []
        for fp in Path(dirt).glob(fmt):
            txts.append(self.load_txt(fp))
        return txts

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        self.stdout(path)
        return obj

    def load_bytes(self, path) -> bytes:
        with open(path, 'rb') as f:
            obj = f.read()

        self.stdout(path)
        return obj

    def load_img(self, path, channel_fixed_3=False) -> np.ndarray:
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        # but still bugs with `cv2.imread`, so still use the following method to read images
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        # img = cv2.imread(path)
        assert img is not None

        if channel_fixed_3:
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

        self.stdout(path)
        return img

    def load_image_from_zipfile(self, path, zip_file):
        from .converter import DataConvert

        image = zip_file.open(path).read()
        image = DataConvert.bytes_to_image(image)

        self.stdout(path)
        return image

    def load_images_from_pdf(
            self,
            obj: str or bytes, scale_ratio: float = 1.33,
            rotate: int = 0, alpha=False, bgr=False
    ) -> List[np.ndarray]:
        import fitz  # pip install PyMuPDF

        if isinstance(obj, str):
            doc = fitz.open(obj)
        else:
            doc = fitz.open(stream=obj, filetype='pdf')

        images = []

        for page in doc:
            trans = fitz.Matrix(scale_ratio, scale_ratio).prerotate(rotate)  # rotate means clockwise
            pm = page.get_pixmap(matrix=trans, alpha=alpha)
            data = pm.tobytes()  # bytes
            img_np = np.frombuffer(data, dtype=np.uint8)

            flag = 1 if bgr else -1
            img = cv2.imdecode(img_np, flags=flag)

            images.append(img)

        if isinstance(obj, str):
            self.stdout(obj)

        return images

    def load_images_from_pdf2(
            self,
            obj: str or bytes, scale_ratio: float = 1.33,
    ) -> List[np.ndarray]:
        # pip install pdf2image
        from pdf2image import convert_from_path, convert_from_bytes, exceptions

        dpi = 72 * scale_ratio
        try:
            if isinstance(obj, (str, Path)):
                images = convert_from_path(obj, dpi=dpi)
            else:
                images = convert_from_bytes(obj, dpi=dpi)
        except exceptions.PDFPageCountError:
            images = []

        images = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]

        if isinstance(obj, str):
            self.stdout(obj)

        return images


saver = Saver()
loader = Loader()


class Cacher:
    def __init__(self, saver=None, loader=None, deleter=None, max_size=None):
        self.saver = saver
        self.loader = loader
        self.deleter = deleter
        self.max_size = max_size

    def __call__(self, *args, **kwargs):
        return self.cache_one(*args, **kwargs)

    def cache_one(self, obj, _id=None):
        self.delete_over_range()
        return self.saver(obj, _id=_id)

    def cache_batch(self, objs, _ids=None):
        _ids = _ids or [None] * len(objs)
        return [self.cache_one(obj, _id=_id) for obj, _id in zip(objs, _ids)]

    def delete_over_range(self):
        if not self.max_size:
            return

        if len(self.saver) >= self.max_size:
            self.deleter()

    def get_one(self, _id=None):
        return self.loader(_id)

    def get_batch(self, _ids=()):
        return [self.get_one(_id) for _id in _ids]


class MemoryCacher:
    def __init__(self, max_size=None,
                 verbose=True, stdout_method=print,
                 **saver_kwargs
                 ):
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.cache = {}

    def __call__(self, *args, **kwargs):
        return self.cache_one(*args, **kwargs)

    def cache_one(self, obj, _id=None, stdout_fmt='Save (%s, %s) successful!'):
        time_str = time.time()
        uid = str(uuid.uuid4())
        if _id is None:
            _id = (uid, time_str)
        self.delete_over_range(stdout_fmt.replace('Save', 'Delete'))
        self.cache[_id] = obj
        self.stdout_method(stdout_fmt % _id)
        return _id

    def cache_batch(self, objs, _ids=None, stdout_fmt='Save (%s, %s) successful!'):
        _ids = _ids or [None] * len(objs)
        ids = []
        for i, obj in enumerate(objs):
            _id = self.cache_one(obj, stdout_fmt=stdout_fmt)
            ids.append(_id)
        return ids

    def delete_over_range(self, stdout_fmt='Delete (%s, %s) successful!'):
        if not self.max_size:
            return

        if len(self.cache) >= self.max_size:
            key = min(self.cache.keys(), key=lambda x: x[1])
            self.cache.pop(key)
            self.stdout_method(stdout_fmt % key)

    def get_one(self, _id=None, stdout_fmt='Get (%s, %s) successful!'):
        if _id is None:
            _id = random.choice(list(self.cache.keys()))
        self.stdout_method(stdout_fmt % _id)
        return self.cache[_id]

    def get_batch(self, _ids=None, size=None, stdout_fmt='Get %s successful!'):
        if _ids is None:
            keys = list(self.cache.keys())
            _ids = np.random.choice(range(len(keys)), size, replace=False)
            _ids = [keys[_] for _ in _ids]

        self.stdout_method(stdout_fmt % str(_ids))
        return [self.cache[_id] for _id in _ids]


class FileCacher:
    def __init__(self, cache_dir=None, max_size=None,
                 verbose=True, stdout_method=print, stdout_fmt='Have deleted %s successful!',
                 saver_kwargs=dict(), loader_kwargs=dict()):
        mk_dir(cache_dir)
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt
        self.saver = Saver(verbose, stdout_method, **saver_kwargs)
        self.loader = Loader(verbose, stdout_method, **loader_kwargs)

    def __call__(self, *args, **kwargs):
        return self.cache_one(*args, **kwargs)

    def get_fn(self, obj=None, _id=None, file_name=None, file_stem=None):
        if _id is not None:
            file_name = _id
        elif file_name is not None:
            file_name = file_name
        elif file_stem is not None:
            file_name = file_stem + auto_suffix(obj)
        else:
            file_name = str(uuid.uuid4()) + auto_suffix(obj)

        return file_name

    def cache_one(self, obj, _id=None, file_name=None, file_stem=None):
        file_name = self.get_fn(obj, _id, file_name, file_stem)
        path = f'{self.cache_dir}/{file_name}'
        self.delete_over_range(suffix=Path(path).suffix)
        self.saver.auto_save(obj, path)
        return file_name

    def cache_batch(self, objs, _ids=None, file_names=None, file_stems=None):
        _ids = _ids or [None] * len(objs)
        file_names = file_names or [None] * len(objs)
        file_stems = file_stems or [None] * len(objs)
        _fns = []
        for obj, _id, file_name, file_stem in zip(objs, _ids, file_names, file_stems):
            file_name = self.get_fn(obj, _id, file_name, file_stem)
            path = f'{self.cache_dir}/{file_name}'
            self.delete_over_range(suffix=Path(path).suffix)
            self.saver.auto_save(obj, path)
            _fns.append(file_name)
        return _fns

    def delete_over_range(self, suffix=''):
        if not self.max_size:
            return

        caches = [str(_) for _ in self.cache_dir.glob(f'*{suffix}')]

        if len(caches) > self.max_size:
            try:
                ctime = [os.path.getctime(fp) for fp in caches]
                min_ctime = min(ctime)
                old_path = caches[ctime.index(min_ctime)]
                os.remove(old_path)
                self.stdout_method(self.stdout_fmt % old_path)
                return old_path

            except FileNotFoundError:
                # todo: if it occur, number of file would be greater than max_size
                self.stdout_method('Two process thread were crashed while deleting file possibly')
                return

    def get_one(self, _id=None, file_name=None):
        file_name = _id or file_name

        if file_name is None:
            caches = [str(_) for _ in self.cache_dir.glob(f'*')]
            path = random.choice(caches)
        else:
            path = f'{self.cache_dir}/{file_name}'

        return self.loader.auto_load(path)

    def get_batch(self, _ids=None, file_names=None, size=None):
        if _ids is None and file_names is None:
            caches = [str(_) for _ in self.cache_dir.glob(f'*')]
            paths = np.random.choice(caches, size, replace=False)
        else:
            file_names = _ids or file_names
            paths = [f'{self.cache_dir}/{file_name}' for file_name in file_names]

        return [self.loader.auto_load(path) for path in paths]


class MongoDBCacher:
    def __init__(self, host='127.0.0.1', port=27017, user=None, password=None, database=None, collection=None,
                 max_size=None, verbose=True, stdout_method=print, stdout_fmt='Save _id[%s] successful!',
                 **mongo_kwargs):
        from pymongo import MongoClient

        self.client = MongoClient(host, port, **mongo_kwargs)
        self.db = self.client[database]
        self.db.authenticate(user, password)
        self.collection = self.db[collection]
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt

    def __call__(self, *args, **kwargs):
        return self.cache_one(*args, **kwargs)

    def cache_one(self, obj: dict, _id=None):
        self.delete_over_range()

        obj['update_time'] = int(time.time())
        if _id is None:
            x = self.collection.insert_one(obj)
            _id = x.inserted_id

        else:
            self.collection.update_one({'_id': _id}, {'$set': obj}, upsert=True)

        self.stdout_method(self.stdout_fmt % _id)
        return _id

    def cache_batch(self, objs, _ids=None):
        self.delete_over_range(len(objs))
        if _ids is None:
            x = self.collection.insert_many(objs)
            _ids = x.inserted_ids

        else:
            for _id, obj in zip(_ids, objs):
                self.collection.update_one({'_id': _id}, {'$set': obj}, upsert=True)

        return _ids

    def delete_over_range(self, batch_size=1):
        if not self.max_size:
            return

        query = self.collection.find()
        if query.count() > self.max_size - batch_size:
            x = query.sort({'update_time': 1}).limit(1)
            self.collection.delete_one(x)

    def get_one(self, _id=None):
        if _id is None:
            return self.collection.find_one()
        else:
            return self.collection.find_one({'_id': _id})

    def get_batch(self, _ids=None, size=None):
        if _ids is None:
            return [self.collection.find_one() for _ in range(size)]
        else:
            return [self.collection.find_one({'_id': _id}) for _id in _ids]


class RedisCacher:
    """todo"""


class ESCacher:
    """todo"""


class PGSQLCacher:
    """todo"""


class FakeIo:
    """a placeholder, empty io method to cheat some functions which must use an io method,
    it means that the method do nothing in fact,
    it is useful to reduce the amounts of code changes

    Examples
    .. code-block:: python

        # save obj
        io_method = open

        # do not save obj
        io_method = FakeIo

        with io_method(fp, 'w', encoding='utf8') as f:
            f.write(obj)
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)

    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args, **kwargs):
        pass


class FakeApp:
    def __init__(self, *args, **kwargs):
        self.config = dict()
        self.conf = dict()
        self.__dict__.update(kwargs)

    def register_blueprint(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        return nullcontext

    def post(self, *args, **kwargs):
        return nullcontext

    def get(self, *args, **kwargs):
        return nullcontext
