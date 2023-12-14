from .SigHan2014 import Loader as Loader_
from .base import DataRegister, DataLoader


class Loader(Loader_):
    """http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

    Data structure:
        .
        ├── Test
        │   ├── SIGHAN15_CSC_TestInput.txt      # id and text, 1100 items
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
        >>> loader = Loader('../data/sighan8csc_release1.0')
        >>> data = loader(set_type=DataRegister.TRAIN, task='A2')
        >>> next(data[0])
    """

    train_task_dict = {
        'A2': 'SIGHAN15_CSC_A2_Training.sgml',
        'B2': 'SIGHAN15_CSC_B2_Training.sgml'
    }

    test_version = 'SIGHAN15'
