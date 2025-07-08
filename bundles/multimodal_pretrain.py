import torch

from data_parse.cv_data_parse.data_augmentation import scale, channel, pixel_perturbation, Apply
from processor import Process, DataHooks
from utils import torch_utils


class DataProcessForQwenVl(DataHooks):
    post_aug = Apply([
        channel.BGR2RGB(),
        scale.Rectangle(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            raise NotImplementedError

        ret.update(self.post_aug(**ret))
        return ret


class QwenVl(Process):
    pretrained: str

    def set_model(self):
        from models.multimodal_pretrain.Qwen2_VL import Model

        with torch.device('meta'):  # fast to init model
            self.model = Model()

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import Qwen2VLTokenizer

        self.tokenizer = Qwen2VLTokenizer.from_pretrained(
            self.vocab_fn,
            self.encoder_fn,
        )

    def from_pretrained(self):
        from models.multimodal_pretrain.Qwen2_VL import WeightLoader

        state_dict = WeightLoader.auto_load(self.pretrained)
        self.model.load_state_dict(state_dict, strict=False, assign=True)

        self.log(f'Loaded pretrain model')

    def get_model_inputs(self, loop_inputs, train=True):
        image = loop_inputs['image']
        text = loop_inputs['text']

        dialog = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": text
                },
            ],
        },
        ]

        r = self.tokenizer.encode_dialog(dialog)
        r = torch_utils.Converter.force_to_tensors(r, self.device)

        return r
