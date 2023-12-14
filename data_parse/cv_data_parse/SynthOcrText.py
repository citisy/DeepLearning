from pathlib import Path
from .base import DataLoader, DataRegister, get_image
from utils import os_lib


class Loader(DataLoader):
    """https://github.com/oh-my-ocr/text_renderer

    Data structure:
        .
        ├── text
        │   └── *.txt
        ├── char.txt
        ├── bg
        │   └── *.png
        ├── font
        │   └── *.tff
        └── font_list.txt

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.MJSynth import DataRegister, CelebALoader as Loader
            from data_parse.cv_data_parse.base import DataVisualizer

            loader = Loader('data/SynthOcrText')
            data = loader(generator=True, image_type=DataRegister.ARRAY)
            r = next(data[0])

            # visual
            DataVisualizer('data/SynthOcrText/visuals', verbose=False, pbar=False)(data[0])
    """
    default_set_type = [DataRegister.MIX]
    text_length = (5, 15)
    font_size = (15, 20)
    max_size = 10000

    def _call(self, max_size=None, image_type=DataRegister.ARRAY, **gen_kwargs):
        from text_renderer import render, effect, corpus, config, layout

        data_dir = Path(self.data_dir)

        text_paths = list(data_dir.glob('text/*.txt'))
        chars_file = data_dir / 'char.txt'
        font_dir = data_dir / 'font'
        font_list_file = data_dir / 'font_list.txt'
        bg_dir = data_dir / 'bg'
        gray = image_type == DataRegister.GRAY_ARRAY

        perspective_transform = config.NormPerspectiveTransformCfg(20, 20, 1.5)

        cfg = config.RenderCfg(
            bg_dir=bg_dir,
            perspective_transform=perspective_transform,
            gray=gray,
            corpus=[
                corpus.CharCorpus(
                    corpus.CharCorpusCfg(
                        text_paths=text_paths,
                        filter_by_chars=True,
                        chars_file=chars_file,
                        length=self.text_length,
                        font_dir=font_dir,
                        font_list_file=font_list_file,
                        font_size=self.font_size,
                    ),
                ),
                corpus.CharCorpus(
                    corpus.CharCorpusCfg(
                        text_paths=text_paths,
                        filter_by_chars=True,
                        chars_file=chars_file,
                        length=self.text_length,
                        font_dir=font_dir,
                        font_list_file=font_list_file,
                        font_size=self.font_size,
                    ),
                ),
            ],
            corpus_effects=[
                effect.Effects([effect.Padding()]),
                effect.NoEffects()
            ],
            layout=layout.extra_text_line.ExtraTextLineLayout(),
            layout_effects=effect.Effects(effect.Line(p=1)),
        )

        render = render.Render(cfg)

        def gen_func():
            i = 0
            while True:
                yield i, render()
                i += 1

        if max_size is not None:
            self.max_size = max_size
        else:
            max_size = self.max_size

        return self.gen_data(gen_func(), max_size=max_size, **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, return_lower=False, **kwargs) -> dict:
        i, (image, text) = obj

        return dict(
            _id=f'{i}_{text}.png',
            image=image,
            text=text
        )

    def __len__(self):
        return self.max_size

    def get_char_list(self):
        loader = os_lib.Loader(verbose=self.verbose, stdout_method=self.stdout_method)
        return loader.load_txt(f'{self.data_dir}/char.txt')
