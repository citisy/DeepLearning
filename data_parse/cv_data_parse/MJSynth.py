from pathlib import Path
from .base import DataLoader, DataRegister, get_image


class Loader(DataLoader):
    """https://www.robots.ox.ac.uk/~vgg/data/text/

    Data structure:
        ./mnt/ramdisk/max/90kDICT32px
        ├── [i]                     # i <= 4000
        │   └── [j]                 # j <= 7
        │       └── *.jpg
        ├── annotation.txt          # 8919273 items
        ├── annotation_train.txt    # 7224612 items
        ├── annotation_test.txt     # 891927 items
        ├── annotation_val.txt      # 802734 items
        ├── imlist.txt
        └── lexicon.txt             # 88172 words

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.MJSynth import DataRegister, CelebALoader as Loader
            from data_parse.cv_data_parse.base import DataVisualizer

            loader = Loader('data/MJSynth')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            DataVisualizer('data/MJSynth/visuals', verbose=False, pbar=False)(data[0])
    """

    _dir = 'mnt/ramdisk/max/90kDICT32px'
    char_list = [chr(i) for i in list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))]    # 0-9 + A-Z + a-z
    lower_char_list = [chr(i) for i in list(range(48, 58)) + list(range(97, 123))]    # 0-9 + a-z

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        if set_type == DataRegister.MIX:
            gen_func = open(f'{self.data_dir}/{self._dir}/annotation.txt', 'r', encoding='utf8')
        else:
            gen_func = open(f'{self.data_dir}/{self._dir}/annotation_{set_type.value}.txt', 'r', encoding='utf8')

        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, return_lower=False, **kwargs) -> dict:
        fp, _ = obj.strip().split(' ')
        image_path = fp.replace('./', f'{self.data_dir}/{self._dir}/')
        image = get_image(image_path, image_type)

        fp = Path(fp)
        text = fp.stem.split('_')[1]

        if return_lower:
            text = text.lower()

        return dict(
            _id=fp.name,
            image=image,
            text=text
        )

    def load_lexicon(self):
        with open(f'{self.data_dir}/{self._dir}/lexicon.txt') as f:
            words = f.read().strip().split('\n')
        return words
