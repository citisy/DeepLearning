import torch.nn.functional as F
from torch import nn

from models.layers import Linear
from models.text_pretrain import XLMRoberta
from utils import torch_utils


class Config(XLMRoberta.Config):
    pass


class WeightConverter(XLMRoberta.WeightConverter):
    @classmethod
    def from_hf(cls, state_dict):
        state_dict = {k.replace('roberta.', ''): v for k, v in state_dict.items()}
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.convert_dict)

        convert_dict = {
            'classifier.dense': 'head.fcn1.linear',
            'classifier.out_proj': 'head.fcn2.linear'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(XLMRoberta.Model):
    def make_head(self, **head_config):
        return Head(self.backbone.out_features, 1)


class Head(nn.Module):
    def __init__(self, in_features, out_features, drop_prob=0):
        super().__init__()
        self.fcn1 = Linear(in_features, in_features, mode='dla', act=nn.Tanh(), drop_prob=drop_prob)
        self.fcn2 = Linear(in_features, out_features, mode='dl')

    def forward(self, hidden_states, normalize=False):
        # take <s> token (equiv. to [CLS])
        x = hidden_states[:, 0, :]
        x = self.fcn1(x)
        x = self.fcn2(x)
        if normalize:
            x = F.sigmoid(x)
        return x
