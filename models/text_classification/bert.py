from ..text_pretrain.bert import *


def convert_hf_weights(state_dict):
    """convert weights from huggingface model to my model

    Usage:
        .. code-block:: python

            from transformers import BertForPreTraining
            state_dict = BertForSequenceClassification.from_pretrained('...')
            state_dict = convert_hf_weights(state_dict)
            Model(...).load_state_dict(state_dict)
    """
    convert_dict = {
        'bert.embeddings.word_embeddings': 'backbone.embedding.token',
        'bert.embeddings.position_embeddings': 'backbone.embedding.position',
        'bert.embeddings.position_ids': 'backbone.embedding.position_ids',
        'bert.embeddings.token_type_embeddings': 'backbone.embedding.segment',
        'bert.embeddings.LayerNorm': 'backbone.embedding.head.0',
        'bert.encoder.layer.{0}.attention.self.query': 'backbone.encode.{0}.res1.fn.to_qkv.0.0',
        'bert.encoder.layer.{0}.attention.self.key': 'backbone.encode.{0}.res1.fn.to_qkv.1.0',
        'bert.encoder.layer.{0}.attention.self.value': 'backbone.encode.{0}.res1.fn.to_qkv.2.0',
        'bert.encoder.layer.{0}.attention.output.dense': 'backbone.encode.{0}.res1.fn.to_out.1.linear',
        'bert.encoder.layer.{0}.attention.output.LayerNorm': 'backbone.encode.{0}.res1.act',
        'bert.encoder.layer.{0}.intermediate.dense': 'backbone.encode.{0}.res2.fn.0.linear',
        'bert.encoder.layer.{0}.output.dense': 'backbone.encode.{0}.res2.fn.1.linear',
        'bert.encoder.layer.{0}.output.LayerNorm': 'backbone.encode.{0}.res2.act',
        'bert.pooler.dense': 'neck.linear',
        'classifier': 'head.fcn'
    }
    state_dict = torch_utils.convert_state_dict(state_dict, convert_dict)

    return state_dict


class Model(nn.Module):
    def __init__(self, vocab_size, sp_tag_dict, out_features=2, bert_config=Config.base):
        super().__init__()

        self.backbone = Bert(vocab_size, sp_tag_dict, **bert_config)
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
