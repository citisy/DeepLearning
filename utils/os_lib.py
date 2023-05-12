import cv2
import json
import os
import numpy as np
import pickle
import time
import psutil
from functools import wraps
from pathlib import Path
from typing import List


def mk_dir(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


class Saver:
    def __init__(self, verbose=True, stdout_method=None, stdout_fmt='Save to %s successful!'):
        self.verbose = verbose
        self.stdout_method = stdout_method or print
        self.stdout_fmt = stdout_fmt

    def stdout(self, path):
        if self.verbose:
            self.stdout_method(self.stdout_fmt % path)

    def auto_save(self, obj, path: str):
        suffix = Path(path).suffix

        if suffix in ('.js', '.json'):
            self.save_json(obj, path)
        elif suffix in ('.txt',):
            self.save_txt(obj, path)
        elif suffix in ('.pkl',):
            self.save_pkl(obj, path)
        elif suffix in ('.png', '.jpg', '.jpeg',):
            self.save_img(obj, path)
        else:
            self.save_bytes(obj, path)

    def save_json(self, obj: dict, path):
        with open(path, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)

        self.stdout(path)

    def save_txt(self, obj: iter, path):
        with open(path, 'w', encoding='utf8') as f:
            f.write('\n'.join(obj))

        self.stdout(path)

    def save_pkl(self, obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

        self.stdout(path)

    def save_bytes(self, obj: bytes, path):
        with open(path, 'wb') as f:
            f.write(obj)

        self.stdout(path)

    def save_img(self, obj: np.ndarray, path):
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        # cv2.imencode('.png', obj)[1].tofile(path)

        cv2.imwrite(path, obj)

        self.stdout(path)


class Loader:
    def __init__(self, verbose=True, stdout_method=None, stdout_fmt='Read %s successful!'):
        self.verbose = verbose
        self.stdout_method = stdout_method or print
        self.stdout_fmt = stdout_fmt

    def stdout(self, path):
        if self.verbose:
            self.stdout_method(self.stdout_fmt % path)

    def auto_load(self, path: str):
        suffix = Path(path).suffix

        if suffix in ('.js', '.json'):
            obj = self.load_json(path)
        elif suffix in ('.txt',):
            obj = self.load_txt(path)
        elif suffix in ('.pkl',):
            obj = self.load_pkl(path)
        elif suffix in ('.png', '.jpg', '.jpeg',):
            obj = self.load_img(path)
        else:
            obj = self.load_bytes(path)

        return obj

    def load_json(self, path) -> dict:
        obj = json.load(path)
        self.stdout(path)

        return obj

    def load_txt(self, path) -> iter:
        with open(path, 'r', encoding='utf8') as f:
            obj = f.read().split('\n')

        self.stdout(path)
        return obj

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

    def load_img(self, path, channel_fixed_3=True) -> np.ndarray:
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

        img = cv2.imread(path)

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

    def load_pdf_to_images(
            self,
            obj: str or bytes, scale_ratio: float = 1.33,
            rotate: int = 0, alpha=False, bgr=False
    ) -> List[np.ndarray]:
        import fitz

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

    def load_pdf_to_images2(
            self,
            obj: str or bytes, scale_ratio: float = 1.33,
    ) -> List[np.ndarray]:
        from pdf2image import convert_from_path, convert_from_bytes

        dpi = 72 * scale_ratio
        if isinstance(obj, str):
            images = convert_from_path(obj, dpi=dpi)
        else:
            images = convert_from_bytes(obj, dpi=dpi)

        images = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]

        if isinstance(obj, str):
            self.stdout(obj)

        return images


class Cache:
    def __init__(self, verbose=True, stdout_method=None, stdout_fmt='Save to %s successful!'):
        self.verbose = verbose
        self.stdout_method = stdout_method
        self.saver = Saver(verbose, stdout_method, stdout_fmt)

    def auto_cache(self, obj, path, max_size=None):
        p = Path(path)
        save_dir = str(p.parent)
        mk_dir(save_dir)
        self.delete_old_file(save_dir, max_size, suffix=p.suffix)

        saver.auto_save(obj, path)

    def delete_old_file(self, save_dir, max_size=None, suffix=''):
        if not max_size:
            return

        caches = [str(_) for _ in Path(save_dir).glob(f'*.{suffix}')]

        if len(caches) > max_size:
            ctime = [os.path.getctime(fp) for fp in caches]
            min_ctime = min(ctime)
            old_path = caches[ctime.index(min_ctime)]
            os.remove(old_path)

            if self.verbose:
                self.stdout_method(f'Have delete {old_path}!')

            return old_path


class PdfOs:
    @classmethod
    def save_pdf_to_img(cls, path, page=None, image_dir=None, scale_ratio=1.33):
        """pdf转成img保存"""
        imgs = loader.load_pdf_to_images2(
            path,
            scale_ratio=scale_ratio
        )

        if isinstance(page, int):
            imgs = [imgs[page]]

        elif isinstance(page, list):
            imgs = [imgs[_] for _ in page]

        image_dir = image_dir or path.replace('pdfs', 'images').replace(Path(path).suffix, '')

        mk_dir(image_dir)

        for i, img in enumerate(imgs):
            saver.save_img(img, f'{image_dir}/{i}.png')

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


class FakeIo:
    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


saver = Saver()
loader = Loader()
cache = Cache()


class Retry:
    print_method = print
    count = 3
    wait = 15

    @classmethod
    def add_try(
            cls,
            error_message='',
            err_type=(ConnectionError, TimeoutError)
    ):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                for i in range(cls.count):
                    try:
                        ret = func(*args, **kwargs)

                    except err_type as e:
                        if i >= cls.count - 1:
                            raise e

                        msg = error_message or f'Something error occur, sleep {cls.wait} seconds, and then retry'

                        cls.print_method(msg)
                        time.sleep(cls.wait)
                        cls.print_method(f'{i + 2}th try!')

                return ret

            return wrap

        return wrap2


class AutoLog:
    print_method = print
    is_simple_log = True
    is_time_log = True
    is_memory_log = True

    @classmethod
    def simple_log(cls, string):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                r = func(*args, **kwargs)
                if cls.is_simple_log:
                    cls.print_method(string)
                return r

            return wrap

        return wrap2

    @classmethod
    def time_log(cls, prefix_string=''):
        def wrap2(func):

            @wraps(func)
            def wrap(*args, **kwargs):
                if cls.is_time_log:
                    st = time.time()
                    r = func(*args, **kwargs)
                    et = time.time()
                    cls.print_method(f'{prefix_string} - elapse[{et - st:.3f}s]!')
                else:
                    r = func(*args, **kwargs)
                return r

            return wrap

        return wrap2

    @classmethod
    def memory_log(cls, prefix_string=''):
        def wrap2(func):

            @wraps(func)
            def wrap(*args, **kwargs):
                if cls.is_memory_log:
                    a = MemoryInfo.get_process_mem_info()
                    r = func(*args, **kwargs)
                    b = MemoryInfo.get_process_mem_info()
                    cls.print_method(f'{prefix_string}\nbefore: {a}\nafter: {b}')
                else:
                    r = func(*args, **kwargs)
                return r

            return wrap

        return wrap2


class MemoryInfo:
    @staticmethod
    def pretty_str(v):
        for suffix in ['b', 'kb', 'mb', 'gb', 'tb']:
            if v > 1024:
                v /= 1024.
            else:
                return f'{v:.2f} {suffix}'

    @classmethod
    def get_process_mem_info(cls, pretty_output=True):
        """
        uss, 进程独立占用的物理内存（不包含共享库占用的内存）
        rss, 该进程实际使用物理内存（包含共享库占用的全部内存）
        vms, 虚拟内存总量
        """
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        info = {
            'pid': str(pid),
            'uss': info.uss,
            'rss': info.rss,
            'vms': info.vms,
        }

        if pretty_output:
            for k, v in info.items():
                if k != 'pid':
                    info[k] = cls.pretty_str(v)

        return info

    @classmethod
    def get_cpu_mem_info(cls, pretty_output=True):
        """
        percent, 实际已经使用的内存占比
        total, 内存总的大小
        available, 还可以使用的内存
        free, 剩余的内存
        used, 已经使用的内存
        """
        info = dict(psutil.virtual_memory()._asdict())
        if pretty_output:
            for k, v in info.items():
                if k != 'percent':
                    info[k] = cls.pretty_str(v)

        return info

    @classmethod
    def get_gpu_mem_info(cls, device=0, pretty_output=True):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info = dict(
            total=info.free,
            used=info.used,
            free=info.free,
        )
        if pretty_output:
            for k, v in info.items():
                info[k] = cls.pretty_str(v)

        return info

    @classmethod
    def get_mem_info(cls, pretty_output=True):
        info = dict(
            process_mem=cls.get_process_mem_info(pretty_output),
            env_mem=cls.get_cpu_mem_info(pretty_output),
            gpu_mem=cls.get_gpu_mem_info(pretty_output),
        )

        return info
