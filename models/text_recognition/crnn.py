import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from ..layers import Conv, Linear, ConvInModule, OutModule
from utils.torch_utils import initialize_layers


class Model(nn.Module):
    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            in_module=None, backbone=None, neck=None, head=None,
            neck_in_features=64, neck_out_features=256,
            char2id={}, max_len=40, pad=0
    ):
        super().__init__()
        self.in_channels = in_ch
        self.input_size = input_size
        self.out_features = out_features + 1  # 1 gives the blank or unknown char
        self.char2id = char2id
        self.id2char = {v: k for k, v in char2id.items()}
        self.max_len = max_len
        self.pad = pad

        self.input = in_module if in_module is not None else nn.Identity()
        self.backbone = backbone if backbone is not None else Backbone(in_ch)
        self.neck = neck if neck is not None else Neck(neck_in_features, neck_out_features)
        self.head = head if head is not None else nn.Linear(2 * neck_out_features, self.out_features + 1)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean')
        initialize_layers(self)

    def embedding(self, context):
        labels = []
        lens = []
        for text in context:
            label = [self.char2id.get(char, 0) for char in text]

            label = label[:self.max_len]
            lens.append(len(label))
            labels.append(torch.tensor(label))

        return labels, lens

    def forward(self, x, true_label=None):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.training:
            loss = self.loss(pred_label=x, true_label=true_label)
            return {'pred': x, 'loss': loss}
        else:
            return self.post_process(x)

    def loss(self, pred_label, true_label):
        device = pred_label.device
        # (b, ) in {w}
        pred_label_lens = torch.full(size=(pred_label.shape[1],), fill_value=pred_label.shape[0], dtype=torch.long, device=device)
        # (w, b, o)
        pred_label = F.log_softmax(pred_label, dim=2)
        # (\sum l, ), (b, ) \in [0, L]
        true_label, true_label_lens = self.embedding(true_label)
        true_label = torch.cat(true_label).to(device).long()
        true_label_lens = torch.tensor(true_label_lens, device=device).long()

        return self.criterion(pred_label, true_label, pred_label_lens, true_label_lens)

    def post_process(self, x):
        preds = x.max(2)[1]  # (w, b)
        preds = preds.permute(1, 0)
        words = []
        for pred in preds:
            word = []
            for p in pred:
                p = int(p)
                if p == 0:
                    continue
                word.append(self.id2char[p])
            words.append(''.join(word))
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

        self.bi_lstm1 = nn.LSTM(hidden_features, out_features, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(2 * out_features, out_features, bidirectional=True)

    def forward(self, x):
        x = self.c(x)
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)

        return x
