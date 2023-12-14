from bs4 import BeautifulSoup
from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """http://ir.itc.ntnu.edu.tw/lre/clp14csc.html

    Data structure:
        .
        ├── Test
        │   ├── CLP14_CSC_TestInput.txt      # id and context, 1062 items
        │   ├── CLP14_CSC_FinalTestSummary.xlsx
        │   └── CLP14_CSC_TestTruth.txt      # id, location and correction
        ├── Tool
        │   ├── clp14csc.jar                 # metric tool
        │   ├── CLP14_Toy_Evaluation.txt     # result from metric tool
        │   ├── CLP14_Toy_Result.txt         # model outputs
        │   └── CLP14_Toy_Truth.txt          # true labels
        └── Training
            ├── B1_training.sgml             # 3095 items
            └── C1_training.sgml             # 342 items

    Usage:
        >>> loader = Loader('../data/clp14csc_release1.1')
        >>> data = loader(set_type=DataRegister.TRAIN, task='B1')
        >>> next(data[0])
    """

    load_data_dict = {
        DataRegister.TRAIN: 'Training',
        DataRegister.TEST: 'Test'
    }

    train_task_dict = {
        'B1': 'B1_training.sgml',
        'C1': 'C1_training.sgml'
    }

    test_version = 'CLP14'

    def _call(self, set_type=DataRegister.TRAIN, task=None, **kwargs):
        """
        task: 'B1', 'C1' or None
        """
        if set_type == DataRegister.TRAIN:
            return self.gen_data(self._parse_train_data(task), set_type=set_type, **kwargs)
        elif set_type == DataRegister.TEST:
            return self.gen_data(self._parse_test_data(), set_type=set_type, **kwargs)

    def _parse_train_data(self, task=None):
        if task:
            fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TRAIN]}/{self.train_task_dict[task]}'
            with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            text = ''
            for train_task_file in self.train_task_dict.values():
                fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TRAIN]}/{train_task_file}'
                with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
                    text += f.read()

        soup = BeautifulSoup(text)

        for essay in soup.find_all('essay'):
            title = essay.get('title')
            for passage in essay.find_all('passage'):
                yield essay, title, passage

    def _parse_test_data(self):
        fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_version}_CSC_TestInput.txt'
        with open(fn, 'r', encoding='utf8') as f:
            a = f.read().strip().split('\n')

        fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_version}_CSC_TestTruth.txt'
        with open(fn, 'r', encoding='utf8') as f:
            b = f.read().strip().split('\n')

        return zip(a, b)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        if set_type == DataRegister.TRAIN:
            return self._get_train_ret(obj)
        elif set_type == DataRegister.TEST:
            return self._get_test_ret(obj)

    def _get_train_ret(self, obj):
        essay, title, passage = obj
        _id = passage.get('id')
        context = passage.text

        locations = []
        wrongs = []
        corrections = []
        for mistake in essay.find_all('mistake', id=_id):
            assert len(mistake.find_all('wrong')) == 1 and len(mistake.find_all('correction')) == 1, f'{_id = }'

            locations.append(int(mistake.get('location')))
            wrongs.append(mistake.find('wrong').text)
            corrections.append(mistake.find('correction').text)

        return dict(
            _id=_id,
            title=title,
            context=context,
            location=locations,
            wrong=wrongs,
            correction=corrections
        )

    def _get_test_ret(self, obj):
        line, y = obj
        pid, context = line.split('\t')
        _id = pid[5:-1]

        y = y.replace(' ', '').split(',')

        assert _id == y[0], f'{_id =}'

        locations = []
        wrongs = []
        corrections = []

        for j in range(1, len(y), 2):
            n = int(y[j])
            if n == 0:
                continue
            locations.append(n)
            wrongs.append(context[n - 1])
            corrections.append(y[j + 1])

        return dict(
            _id=_id,
            context=context,
            location=locations,
            wrong=wrongs,
            correction=corrections
        )
