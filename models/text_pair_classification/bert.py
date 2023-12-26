from ..text_pretrain.bert import Bert, Config
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, seq_len, sp_tag_dict, bert_config=Config.base):
        super().__init__()

        self.bert = Bert(vocab_size, seq_len, sp_tag_dict, n_segment=2, **bert_config)
        self.head = nn.Linear(self.bert.out_features, 2)

    def forward(self, x, true=None):
        x = self.bert(x)
        x = self.head(x[:, 0])  # select the first output

        losses = {}
        if self.training:
            losses = self.loss(x, true)

        return dict(
            pred=x,
            **losses
        )

    def loss(self, pred, true):
        return F.cross_entropy(pred, true)
