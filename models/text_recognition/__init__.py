import torch
from torch import nn
import torch.nn.functional as F
from utils.torch_utils import initialize_layers


class BaseTextRecModel(nn.Module):
    """a template to make a image classifier model by yourself"""

    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            char2id={}, max_seq_len=25,
            in_module=None, backbone=None, neck=None, head=None,
            neck_out_features=None
    ):
        super().__init__()

        self.in_channels = in_ch
        self.input_size = input_size
        self.out_features = out_features + 1  # 1 gives the blank or unknown char
        self.max_seq_len = max_seq_len
        self.char2id = char2id
        self.id2char = {v: k for k, v in char2id.items()}

        self.input = in_module if in_module is not None else nn.Identity()
        self.backbone = backbone
        self.neck = neck
        self.head = head if head is not None else nn.Linear(neck_out_features, self.out_features)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean')
        initialize_layers(self)

    def forward(self, x, true_label=None):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = F.log_softmax(x, dim=2)

        if self.training:
            loss = self.loss(pred_label=x, true_label=true_label)
            return {'pred': x, 'loss': loss}
        else:
            return self.post_process(x)

    def loss(self, pred_label, true_label):
        device = pred_label.device
        # (b, ) in {w}
        pred_label_lens = torch.full(size=(pred_label.shape[1],), fill_value=pred_label.shape[0], dtype=torch.long, device=device)
        # (\sum l, ), (b, ) \in [0, L]
        true_label, true_label_lens = self.embedding(true_label)
        true_label = torch.cat(true_label).to(device).long()
        true_label_lens = torch.tensor(true_label_lens, device=device).long()

        return self.criterion(pred_label, true_label, pred_label_lens, true_label_lens)

    def embedding(self, context):
        labels = []
        lens = []
        for text in context:
            label = [self.char2id.get(char, 0) for char in text[:self.max_seq_len]]
            lens.append(len(label))
            labels.append(torch.tensor(label))

        return labels, lens
