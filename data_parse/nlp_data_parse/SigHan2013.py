from bs4 import BeautifulSoup
from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html

    Data structure:
        .
        ├── ConfusionSet
        │   ├── Bakeoff2013_CharacterSet_SimilarPronunciation.txt   #
        │   └── Bakeoff2013_CharacterSet_SimilarShape.txt
        ├── EvaluationTool
        │   ├── sighan7csc.jar
        │   ├── Toy_SubTask*_Evaluation.txt
        │   ├── Toy_SubTask*_Result.txt
        │   ├── Toy_SubTask*_Truth.txt
        ├── FinalTest
        │   ├── FinalTest_SubTask1_Truth.txt
        │   ├── FinalTest_SubTask1.txt
        │   ├── FinalTest_SubTask2_Truth.txt
        │   └── FinalTest_SubTask2.txt
        └── SampleSet
            ├── Bakeoff2013_SampleSet_WithError_00001-00350.txt         # 350 items
            └── Bakeoff2013_SampleSet_WithoutError_10001-10350.txt      # 350 items

    Usage:
        >>> loader = Loader('../data/sighan7csc_release1.0')
        >>> data = loader(set_type=DataRegister.TRAIN, task='with_error')
        >>> next(data[0])
    """

    load_data_dict = {
        DataRegister.TRAIN: 'SampleSet',
        DataRegister.TEST: 'FinalTest'
    }

    train_task_dict = {
        'with_error': 'Bakeoff2013_SampleSet_WithError_00001-00350.txt',
        'without_error': 'Bakeoff2013_SampleSet_WithoutError_10001-10350.txt'
    }

    test_task_dict = {
        '1': 'FinalTest_SubTask1',
        '2': 'FinalTest_SubTask2'
    }

    def _call(self, set_type=DataRegister.TRAIN, task=None, test_task=None, **kwargs):
        if set_type == DataRegister.TRAIN:
            return self.gen_data(self._parse_train_data(task), set_type=set_type, **kwargs)
        elif set_type == DataRegister.TEST:
            return self.gen_data(self._parse_test_data(test_task), set_type=set_type, test_task=test_task, **kwargs)

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
        return soup.find_all('doc')

    def _parse_test_data(self, test_task=None):
        tmp = {}
        # read file
        if test_task:
            fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_task_dict[test_task]}.txt'
            with open(fn, 'r', encoding='utf8', errors='ignore') as f:
                a = f.read().strip().split('\n')

            fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_task_dict[test_task]}_Truth.txt'
            with open(fn, 'r', encoding='utf8', errors='ignore') as f:
                b = f.read().strip().split('\n')

            tmp[test_task] = (a, b)
        else:
            for test_task, test_task_file in self.test_task_dict.items():
                fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{test_task_file}.txt'
                with open(fn, 'r', encoding='utf8', errors='ignore') as f:
                    a = f.read().strip().split('\n')

                fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{test_task_file}_Truth.txt'
                with open(fn, 'r', encoding='utf8', errors='ignore') as f:
                    b = f.read().strip().split('\n')

                tmp[test_task] = (a, b)

        for test_task, (a, b) in tmp.items():
            for i, line in enumerate(a):
                yield line, b[i]

    def get_ret(self, obj, set_type=DataRegister.TRAIN, test_task=None, **kwargs) -> dict:
        if set_type == DataRegister.TRAIN:
            return self._get_train_ret(obj)
        elif set_type == DataRegister.TEST:
            return self._get_test_ret(obj, test_task=test_task)

    def _get_train_ret(self, obj):
        essay = obj
        _id = essay.get('nid')
        text = essay.text

        locations = []
        wrongs = []
        corrections = []
        for mistake in essay.find_all('mistake'):
            locations.append(int(mistake.get('wrong_position')))

            wrongs = mistake.find_all('wrong')
            corrects = mistake.find_all('correct')
            assert len(wrongs) == len(corrects), f'{_id = }'

            if len(wrongs) == 0:
                continue

            wrongs.append(mistake.find('wrong').text)
            corrections.append(mistake.find('correct').text)

        return dict(
            _id=_id,
            text=text,
            location=locations,
            wrong=wrongs,
            correction=corrections
        )

    def _get_test_ret(self, obj, test_task=None):
        line, y = obj
        pid, text = line.strip().split(' ', 1)
        _id = pid[5:-1]

        y = y.replace(' ', '').split(',')

        assert _id == y[0], f'{_id =}'

        locations = []
        wrongs = []
        corrections = []

        if test_task == '1':
            for j in range(1, len(y)):
                if not y[j]:
                    continue
                n = int(y[j])
                if n == 0:
                    continue
                locations.append(n)
                wrongs.append(text[n - 1])
                corrections.append('')

        elif test_task == '2':
            for j in range(1, len(y), 2):
                if not y[j]:
                    continue
                n = int(y[j])
                if n == 0:
                    continue
                locations.append(n)
                wrongs.append(text[n - 1])
                corrections.append(y[j + 1])
        else:
            raise ValueError(f'Unknown input {test_task = }')

        return dict(
            _id=_id,
            text=text,
            location=locations,
            wrong=wrongs,
            correction=corrections
        )
