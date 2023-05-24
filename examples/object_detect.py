from .base import OdProcess


class FastererRCNN_Voc(OdProcess):
    """
    Usage:
        .. code-block:: python
            from examples.object_detect import FastererRCNN_Voc as Process

            Process().run()
            {'score': 0.9899}
    """

    def __init__(self):
        from object_detection.FasterRCNN import Model

        in_ch = 3
        input_size = 800
        output_size = 20

        super().__init__(
            model=Model(
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
                output_size=output_size
            ),
            model_version='FastererRCNN',
            dataset_version='Voc2012',
            input_size=input_size,
            device=1
        )

