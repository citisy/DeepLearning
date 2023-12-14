from .base import DataRegister, DataLoader, DataSaver, get_image

info = [
    {
        'url': 'http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip',
        'fp': 'notredame.zip',
        'md5': '509eda8535847b8c0a90bbb210c83484',
        'len': 468159,
        'mean': 0.4854,
        'std': 0.1864
    },

    {
        'url': 'http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip',
        'fp': 'yosemite.zip',
        'md5': '533b2e8eb7ede31be40abc317b2fd4f0',
        'len': 633587,
        'mean': 0.4844,
        'std': 0.1818
    },

    {
        'url': 'http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip',
        'fp': 'liberty.zip',
        'md5': 'fdd9152f138ea5ef2091746689176414',
        'len': 450092,
        'mean': 0.4437,
        'std': 0.2019
    },

    {
        'url': 'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
        'fp': 'notredame_harris.zip',
        'md5': '69f8c90f78e171349abdf0307afefe4d',
        'len': 379587,
        'mean': 0.4854,
        'std': 0.1864
    },

    {
        'url': 'http://matthewalunbrown.com/patchdata/yosemite_harris.zip',
        'fp': 'yosemite_harris.zip',
        'md5': 'a73253d1c6fbd3ba2613c45065c00d46',
        'len': 450912,
        'mean': 0.4844,
        'std': 0.1818
    },

    {
        'url': 'http://matthewalunbrown.com/patchdata/liberty_harris.zip',
        'fp': 'liberty_harris.zip',
        'md5': 'c731fcfb3abb4091110d0ae8c7ba182c',
        'len': 325295,
        'mean': 0.4437,
        'std': 0.2019
    }
]


class Loader(DataLoader):
    """[Learning Local Image Descriptors Data](http://phototour.cs.washington.edu/patches/default.htm)

    """

    dataset_info = info

    image_suffix = 'bmp'
