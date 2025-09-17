import torch
from torch import nn
import torch.nn.functional as F
from utils.torch_utils import ModuleManager


class BaseTextRecModel(nn.Module):
    """a template to make a image classifier model by yourself"""

    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            char2id=None, id2char=None, max_seq_len=25,
            in_module=None, backbone=None, neck=None, head=None,
            neck_out_features=None
    ):
        super().__init__()

        self.in_channels = in_ch
        self.input_size = input_size
        self.out_features = out_features
        self.max_seq_len = max_seq_len

        if not char2id:
            char2id = {v: k for k, v in id2char.items()}
        self.char2id = char2id

        if not id2char:
            id2char = {v: k for k, v in char2id.items()}
        self.id2char = id2char

        self.input = in_module if in_module is not None else nn.Identity()
        self.backbone = backbone
        self.neck = neck
        self.head = head if head is not None else nn.Linear(neck_out_features, self.out_features)
        # ModuleManager.initialize_layers(self)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, true_label=None):
        x = self.process(x)
        loss = self.loss(pred_label=x, true_label=true_label)
        return {'pred': x, 'loss': loss}

    def inference(self, x):
        x = self.process(x)
        return self.post_process(x)

    def process(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=2)
        return x

    def loss(self, pred_label, true_label):
        device = pred_label.device
        # (b, ) in {w}
        pred_label_lens = torch.full(size=(pred_label.shape[1],), fill_value=pred_label.shape[0], dtype=torch.long, device=device)
        # (\sum l, ), (b, ) \in [0, L]
        true_label, true_label_lens = self.embedding(true_label)
        true_label = torch.cat(true_label).to(device).long()
        true_label_lens = torch.tensor(true_label_lens, device=device).long()

        return F.ctc_loss(pred_label, true_label, pred_label_lens, true_label_lens, blank=self.char2id[' '], reduction='mean')

    def embedding(self, context):
        labels = []
        lens = []
        for text in context:
            label = [self.char2id.get(char, self.char2id[' ']) for char in text[:self.max_seq_len]]
            lens.append(len(label))
            labels.append(torch.tensor(label))

        return labels, lens
