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
    BASE64 = 3
