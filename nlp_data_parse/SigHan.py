from bs4 import BeautifulSoup
from .base import DataRegister, DataLoader


class SigHan2013Loader(DataLoader):
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
        >>> loader = SigHan2013Loader('../data/sighan7csc_release1.0')
        >>> data = loader(data_type=DataRegister.TRAIN, task='with_error')
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

    def _call(self, load_type, task=None, test_task=None, **kwargs):
        if load_type == DataRegister.TRAIN:
            return self.load_train(task)
        elif load_type == DataRegister.TEST:
            return self.load_test(test_task)

    def load_train(self, task):
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

        for essay in soup.find_all('doc'):
            _id = essay.get('nid')
            context = essay.text

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

            yield dict(
                _id=_id,
                context=context,
                location=locations,
                wrong=wrongs,
                correction=corrections
            )

    def load_test(self, test_task):
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

        # parse
        for test_task, (a, b) in tmp.items():
            for i, line in enumerate(a):
                pid, context = line.strip().split(' ', 1)
                _id = pid[5:-1]

                y = b[i].replace(' ', '').split(',')

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
                        wrongs.append(context[n - 1])
                        corrections.append('')

                elif test_task == '2':
                    for j in range(1, len(y), 2):
                        if not y[j]:
                            continue
                        n = int(y[j])
                        if n == 0:
                            continue
                        locations.append(n)
                        wrongs.append(context[n - 1])
                        corrections.append(y[j + 1])
                else:
                    raise ValueError(f'Unknown input {test_task = }')

                yield dict(
                    _id=_id,
                    context=context,
                    location=locations,
                    wrong=wrongs,
                    correction=corrections
                )


class SigHan2014Loader(DataLoader):
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
        >>> loader = SigHan2014Loader('../data/clp14csc_release1.1')
        >>> data = loader(data_type=DataRegister.TRAIN, task='B1')
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

    def _call(self, load_type, task=None, **kwargs):
        """
        task: 'B1', 'C1' or None
        """
        if load_type == DataRegister.TRAIN:
            return self.load_train(task)
        elif load_type == DataRegister.TEST:
            return self.load_test()

    def load_train(self, task=None, ):
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

                yield dict(
                    _id=_id,
                    title=title,
                    context=context,
                    location=locations,
                    wrong=wrongs,
                    correction=corrections
                )

    def load_test(self):
        fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_version}_CSC_TestInput.txt'
        with open(fn, 'r', encoding='utf8') as f:
            a = f.read().strip().split('\n')

        fn = f'{self.data_dir}/{self.load_data_dict[DataRegister.TEST]}/{self.test_version}_CSC_TestTruth.txt'
        with open(fn, 'r', encoding='utf8') as f:
            b = f.read().strip().split('\n')

        for i, line in enumerate(a):
            pid, context = line.split('\t')
            _id = pid[5:-1]

            y = b[i].replace(' ', '').split(',')

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

            yield dict(
                _id=_id,
                context=context,
                location=locations,
                wrong=wrongs,
                correction=corrections
            )


class SigHan2015Loader(SigHan2014Loader):
    """http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

    Data structure:
        .
        ├── Test
        │   ├── SIGHAN15_CSC_TestInput.txt      # id and context, 1100 items
        │   ├── SIGHAN15_CSC_TestSummary.xlsx
        │   └── SIGHAN15_CSC_TestTruth.txt      # id, location and correction
        ├── Tool
        │   ├── sighan15csc.jar                 # metric tool
        │   ├── SIGHAN15_Toy_Evaluation.txt     # result from metric tool
        │   ├── SIGHAN15_Toy_Result.txt         # model outputs
        │   └── SIGHAN15_Toy_Truth.txt          # true labels
        └── Training
            ├── SIGHAN15_CSC_A2_Training.sgml   # 835 items
            └── SIGHAN15_CSC_B2_Training.sgml   # 1504 items

    Usage:
        >>> loader = SigHan2015Loader('../data/sighan8csc_release1.0')
        >>> data = loader(data_type=DataRegister.TRAIN, task='A2')
        >>> next(data[0])
    """

    train_task_dict = {
        'A2': 'SIGHAN15_CSC_A2_Training.sgml',
        'B2': 'SIGHAN15_CSC_B2_Training.sgml'
    }

    test_version = 'SIGHAN15'


class Wang271kLoader(DataLoader):
    """https://github.com/wdimmy/Automatic-Corpus-Generation

    Data structure:
        .
        └── train.sgml  # 271329 items
    """

    def __call__(self, data_type=DataRegister.MIX, generator=True, **kwargs):
        assert data_type == DataRegister.MIX, f'data_type only support `Register.MIX`'
        return super(Wang271kLoader, self).__call__(data_type, generator, **kwargs)

    def _call(self, load_type, **kwargs):
        fn = f'{self.data_dir}/train.sgml'
        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        soup = BeautifulSoup(text, 'lxml')

        for i, essay in enumerate(soup.find_all('sentence')):
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

            yield dict(
                _id=i,
                context=context,
                location=locations,
                wrong=wrongs,
                correction=corrections
            )


Loader = SigHan2013Loader
