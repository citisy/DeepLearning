from enum import Enum


class DataRegister(Enum):
    place_holder = None

    # set type
    MIX = 'mix'
    FULL = 'full'
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    DEV = 'dev'
    TRAIN_VAL = 'trainval'

    # image type
    PATH = 1
    ARRAY = 2
    GRAY_ARRAY = 2.1
    NPY = 2.2
    BASE64 = 3
    Zip = 4
