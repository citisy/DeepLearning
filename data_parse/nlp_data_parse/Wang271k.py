from .base import DataRegister, DataLoader
from bs4 import BeautifulSoup


class Loader(DataLoader):
    """https://github.com/wdimmy/Automatic-Corpus-Generation

    Data structure:
        .
        └── train.sgml  # 271329 items
    """

    def __call__(self, set_type=DataRegister.MIX, generator=True, **kwargs):
        assert set_type == DataRegister.MIX, f'set_type only support `Register.MIX`'
        return super().__call__(set_type, generator, **kwargs)

    def _call(self, **kwargs):
        fn = f'{self.data_dir}/train.sgml'
        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        soup = BeautifulSoup(text, 'lxml')
        return self.gen_data(enumerate(soup.find_all('sentence')), **kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        i, essay = obj
        passage = essay.find('text')
        context = passage.text

        locations = []
        wrongs = []
        corrections = []
        for mistake in passage.find_all('mistake'):
            assert len(mistake.find_all('wrong')) == 1 and len(mistake.find_all('correction')) == 1, f'{i = }'

            locations.append(int(mistake.get('location')))
            wrongs.append(mistake.find('wrong').text)
            corrections.append(mistake.find('correction').text)

        return dict(
            _id=i,
            context=context,
            location=locations,
            wrong=wrongs,
            correction=corrections
        )
