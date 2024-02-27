import torch
from torch import nn, einsum
from utils import torch_utils
from . import ldm


class Config(ldm.Config):
    """only for inference"""
    # support version v1, v1.*

    # for CLIPEmbedder layer output
    LAST = 'last'
    HIDDEN = 'hidden'
    POOLED = 'pooled'


class Model(ldm.Model):
    """
    https://github.com/CompVis/stable-diffusion
    """

    def make_cond(self, cond_config=dict(), **kwargs):
        return CLIPEmbedder(**cond_config)


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)
    see https://huggingface.co/openai/clip-vit-large-patch14"""

    def __init__(self, pretrain_model=None, load_weight=False, layer=Config.LAST, layer_idx=None, return_pooled=False):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

        pretrain_model = pretrain_model or 'openai/clip-vit-large-patch14'
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrain_model)

        if load_weight:
            self.transformer = CLIPTextModel.from_pretrained(pretrain_model)

        else:
            # if having ldm pretrain_model, do not download the clip weight file, only the config file
            # 'cause the ldm pretrain_model contains the clip weight
            configuration = CLIPTextConfig.from_pretrained(pretrain_model)
            self.transformer = CLIPTextModel(configuration)
        self.max_length = self.tokenizer.model_max_length  # 77
        self.output_size = self.transformer.config.hidden_size  # 768
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = return_pooled

    def forward(self, text):
        if isinstance(text, torch.Tensor):
            tokens = text
        else:
            tokens = self.tokenize(text).to(self.transformer.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == Config.HIDDEN)

        if self.layer == Config.LAST:
            z = outputs.last_hidden_state
        elif self.layer == Config.POOLED:
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]

        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def tokenize(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        return batch_encoding["input_ids"]
