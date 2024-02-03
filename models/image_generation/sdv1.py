from torch import nn, einsum
from utils import torch_utils
from . import ldm
from .ldm import convert_weights


class Config(ldm.Config):
    in_module = dict(
        pretrain_model='openai/clip-vit-large-patch14'
    )


class Model(ldm.Model):
    """
    https://github.com/CompVis/stable-diffusion
    """
    def __init__(self, *args, in_module_config=Config.in_module, **kwargs):
        in_module = CLIPEmbedder(**in_module_config)

        super().__init__(
            *args,
            in_module=in_module,
            **kwargs
        )


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)
    see https://huggingface.co/openai/clip-vit-large-patch14"""

    def __init__(self, pretrain_model=None, max_length=77, load_weight=False):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

        pretrain_model = pretrain_model or 'openai/clip-vit-large-patch14'
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrain_model)

        if load_weight:
            self.transformer = CLIPTextModel.from_pretrained(pretrain_model)

        else:
            configuration = CLIPTextConfig.from_pretrained(pretrain_model)
            self.transformer = CLIPTextModel(configuration)
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.transformer.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z
