from ..text_pretrain.bert import *


class Model(nn.Module):
    def __init__(self, vocab_size, pad_id, out_features=2, bert_config=Config.get()):
        super().__init__()

        self.backbone = Bert(vocab_size, pad_id, **bert_config)
        self.neck = Linear(self.backbone.out_features, self.backbone.out_features, mode='lad', act=nn.Tanh(), drop_prob=0.1)
        self.head = ModelForSeqCls(self.backbone.out_features, out_features)

    def forward(self, x, segment_label, attention_mask=None, seq_cls_true=None, **kwargs):
        x = self.backbone(x, segment_label=segment_label, attention_mask=attention_mask)

        seq_cls_logit = self.head(x)
        outputs = {'seq_cls_logit': seq_cls_logit}

        losses = {}
        if self.training:
            losses = self.loss(seq_cls_logit, seq_cls_true)

        outputs.update(losses)
        return outputs

    def loss(self, seq_cls_logit, seq_cls_true):
        next_loss = self.head.loss(seq_cls_logit, seq_cls_true)
        return {'loss': next_loss}
