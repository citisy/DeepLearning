from .base import OdProcess


class FastererRCNN_Voc(OdProcess):
    def __init__(self):
        from object_detection.FasterRCNN import Model

        in_ch = 3
        input_size = 500
        output_size = 20

        super().__init__(
            model=Model(in_ch, input_size, output_size),
            model_version='AlexNet',
            input_size=input_size,
            device=0
        )
