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
        # 'bert.embeddings.position_embeddings.weight': 'backbone.embedding.position.pe',
        'bert.embeddings.token_type_embeddings': 'backbone.embedding.segment',
        'bert.embeddings.LayerNorm': 'backbone.embedding.head.0',
        'bert.encoder.layer.{0}.attention.self.query': 'backbone.encode.{0}.attention.to_qkv.0.0',
        'bert.encoder.layer.{0}.attention.self.key': 'backbone.encode.{0}.attention.to_qkv.1.0',
        'bert.encoder.layer.{0}.attention.self.value': 'backbone.encode.{0}.attention.to_qkv.2.0',
        'bert.encoder.layer.{0}.attention.output.dense': 'backbone.encode.{0}.attention.to_out.1',
        'bert.encoder.layer.{0}.intermediate.dense': 'backbone.encode.{0}.feed_forward.0.linear',
        'bert.encoder.layer.{0}.output.dense': 'backbone.encode.{0}.feed_forward.1.linear',
        'bert.encoder.layer.{0}.attention.output.LayerNorm': 'backbone.encode.{0}.res1.norm',
        'bert.encoder.layer.{0}.output.LayerNorm': 'backbone.encode.{0}.res2.norm',
        'bert.pooler.dense': 'neck.linear'
    }
    state_dict = torch_utils.convert_state_dict(state_dict, convert_dict)

    return state_dict


class Model(nn.Module):
    def __init__(self, vocab_size, sp_tag_dict, out_features=2, bert_config=Config.base):
        super().__init__()

        self.backbone = Bert(vocab_size, sp_tag_dict, **bert_config)
        self.neck = Linear(self.backbone.out_features, self.backbone.out_features, mode='la', act=nn.Tanh())
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
