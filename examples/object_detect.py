from .base import OdProcess
from cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, channel, RandomApply, Apply, complex
import numpy as np


class FastererRCNN_Voc(OdProcess):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import FastererRCNN_Voc as Process

            Process().run()
            {'score': 0.3910}
    """

    def __init__(self):
        from object_detection.FasterRCNN import Model

        in_ch = 3
        input_size = 800
        n_classes = 20

        super().__init__(
            model=Model(
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
                n_classes=n_classes
            ),
            model_version='FastererRCNN',
            dataset_version='Voc2012',
            input_size=input_size,
            device=1
        )


class YoloV5_Voc(OdProcess):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import YoloV5_Voc as Process

            Process().run(max_epoch=500)
            {'score': 0.6721}
    """

    def __init__(self):
        from object_detection.YoloV5 import Model

        in_ch = 3
        input_size = 640
        n_classes = 20

        super().__init__(
            model=Model(
                n_classes,
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
            ),
            model_version='YoloV5',
            dataset_version='Voc2012',
            input_size=input_size,
            device=1
        )

    def data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        return ret

    def complex_data_augment(self, idx, data, base_process):
        if np.random.random() > 0.5:
            idxes = [idx] + list(np.random.choice(range(len(data)), 3, replace=False))
            rets = []
            for idx in idxes:
                ret = base_process(idx)
                rets.append(ret)

            image_list = [ret['image'] for ret in rets]
            bboxes_list = [ret['bboxes'] for ret in rets]
            classes_list = [ret['classes'] for ret in rets]
            img_size = np.max([img.shape[:2] for img in image_list])
            ret = complex.Mosaic(img_size=img_size)(
                image_list=image_list,
                bboxes_list=bboxes_list,
                classes_list=classes_list
            )
        else:
            ret = base_process(idx)

        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            pixel_perturbation.MinMax(),
            pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))

        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            pixel_perturbation.MinMax(),
            pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret
