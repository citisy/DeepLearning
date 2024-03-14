from ..text_pretrain.bert import *


class Model(nn.Module):
    def __init__(self, vocab_size, pad_id, out_features=2, bert_config=Config.get()):
        super().__init__()

        self.backbone = Bert(vocab_size, pad_id, **bert_config)
        self.neck = Linear(self.backbone.out_features, self.backbone.out_features, mode='lad', act=nn.Tanh(), drop_prob=0.1)
        self.head = NSP(self.backbone.out_features, out_features)

    def forward(self, x, segment_label, attention_mask=None, next_true=None, **kwargs):
        x = self.backbone(x, segment_label, attention_mask)

        next_pred = self.head(x)
        outputs = {'next_pred': next_pred}

        losses = {}
        if self.training:
            losses = self.loss(next_pred, next_true)

        outputs.update(losses)
        return outputs

    def loss(self, next_pred, next_true):
        next_loss = self.head.loss(next_pred, next_true)
        return {'loss': next_loss}
