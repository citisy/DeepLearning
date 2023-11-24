import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from ..layers import Conv, Linear, ConvInModule, OutModule
from . import BaseTextRecModel
from data_parse.nlp_data_parse.pre_process import Decoder


class Model(BaseTextRecModel):
    """refer to: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)"""
    def __init__(self, in_ch=3, neck_in_features=64, neck_out_features=512, **kwargs):
        super().__init__(
            backbone=Backbone(in_ch),
            neck=Neck(neck_in_features, neck_out_features),
            in_ch=in_ch,
            neck_out_features=neck_out_features,
            **kwargs
        )

    def post_process(self, x):
        x = x.permute(1, 0, 2)
        preds, probs = Decoder.beam_search(x, beam_size=10)
        words = []
        for b in range(x.shape[0]):
            seq = {}
            for pred, prob in zip(preds[b], probs[b]):
                # note that, filter the duplicate chars can raise the score obviously,
                # but it would filter the right result while there are duplicate chars in the true labels
                diff = torch.diff(pred)
                diff = torch.cat([torch.tensor([-1]).to(diff), diff])
                pred = pred[diff != 0]
                pred = pred[pred != 0]
                pred = tuple(pred)
                seq[pred] = torch.log(torch.exp(prob) + (torch.exp(seq[pred]) if pred in seq else 0))

            chars = max(seq.items(), key=lambda x: x[1])[0]
            chars = [self.id2char[int(c)] for c in chars]
            words.append(''.join(chars))

        return {'pred': words}


class Backbone(nn.Sequential):
    def __init__(self, in_ch):
        layers = [
            Conv(in_ch, 64, 3, is_norm=False),
            nn.MaxPool2d(2),

            Conv(64, 128, 3, is_norm=False),
            nn.MaxPool2d(2),

            Conv(128, 256, 3, is_norm=False),
            Conv(256, 256, 3, is_norm=False),
            nn.MaxPool2d((2, 1)),

            Conv(256, 512, 3),
            Conv(512, 512, 3),
            nn.MaxPool2d((2, 1)),

            Conv(512, 512, 2, p=0, is_norm=False)
        ]

        super().__init__(*layers)


class Neck(nn.Module):
    def __init__(self, hidden_features, out_features):
        super().__init__()

        self.c = nn.Sequential(
            Rearrange('b c h w -> w b (c h)'),
            nn.LazyLinear(hidden_features),
        )

        self.bi_lstm1 = nn.LSTM(hidden_features, out_features // 2, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(out_features, out_features // 2, bidirectional=True)

    def forward(self, x):
        x = self.c(x)
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)

        return x
