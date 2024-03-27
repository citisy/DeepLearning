import math
import torch
import torch.nn.functional as F
from torch import nn
from utils import torch_utils
from .transformers import EncoderEmbedding, TransformerSequential
from .. import bundles
from ..layers import Linear


class Config(bundles.Config):
    default_model = 'base'

    @classmethod
    def make_full_config(cls) -> dict:
        config_dict = dict(
            tiny=dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2),
            mini=dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4),
            small=dict(hidden_size=512, num_hidden_layers=4, num_attention_heads=8),
            medium=dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8),
            base=dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
            large=dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16),
        )
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def auto_download(cls, save_path, save_name=''):
        # download weight auto from transformers
        from transformers import BertForMaskedLM

        model = BertForMaskedLM.from_pretrained(save_path, num_labels=2)
        state_dict = model.state_dict()
        return state_dict


class WeightConverter:
    @staticmethod
    def from_hf(state_dict):
        """convert weights from huggingface model to my own model

        Usage:
            .. code-block:: python

                state_dict = WeightLoader.from_hf(...)
                state_dict = WeightConverter.from_hf(state_dict)
                Model(...).load_state_dict(state_dict)

        """
        convert_dict = {
            'bert.embeddings.word_embeddings': 'backbone.embedding.token',
            'bert.embeddings.position_embeddings': 'backbone.embedding.position',
            'bert.embeddings.position_ids': 'backbone.embedding.position_ids',
            'bert.embeddings.token_type_embeddings': 'backbone.embedding.segment',
            'bert.embeddings.LayerNorm': 'backbone.embedding.head.0',
            'bert.encoder.layer.{0}.attention.self.query': 'backbone.encoder.{0}.attn_res.fn.to_qkv.0',
            'bert.encoder.layer.{0}.attention.self.key': 'backbone.encoder.{0}.attn_res.fn.to_qkv.1',
            'bert.encoder.layer.{0}.attention.self.value': 'backbone.encoder.{0}.attn_res.fn.to_qkv.2',
            'bert.encoder.layer.{0}.attention.output.dense': 'backbone.encoder.{0}.attn_res.fn.to_out.linear',
            'bert.encoder.layer.{0}.attention.output.LayerNorm': 'backbone.encoder.{0}.attn_res.norm',
            'bert.encoder.layer.{0}.intermediate.dense': 'backbone.encoder.{0}.ff_res.fn.0.linear',
            'bert.encoder.layer.{0}.output.dense': 'backbone.encoder.{0}.ff_res.fn.1.linear',
            'bert.encoder.layer.{0}.output.LayerNorm': 'backbone.encoder.{0}.ff_res.norm',
            'bert.pooler.dense': 'neck.linear',
            'cls.predictions.transform.dense': 'mlm.0.linear',
            'cls.predictions.transform.LayerNorm': 'mlm.0.norm',
            'cls.predictions.decoder': 'mlm.1',
            'cls.predictions.bias': 'mlm.1.bias'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        # convert the original weight
        convert_dict = {
            '{0}.gamma': '{0}.weight',
            '{0}.beta': '{0}.bias'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    """refer to
    paper:
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    code:
        - https://github.com/google-research/bert
        - https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert
        - https://github.com/codertimo/BERT-pytorch
    """

    def __init__(
            self, vocab_size, pad_id, skip_id,
            bert_config=Config.get('base'),
            is_nsp=True, nsp_out_features=2,
            is_mlm=True
    ):
        super().__init__()
        self.is_nsp = is_nsp
        self.is_mlm = is_mlm

        self.backbone = Bert(vocab_size, pad_id, **bert_config)
        if is_nsp:
            self.nsp = NSP(self.backbone.out_features, nsp_out_features)
        if is_mlm:
            self.mlm = MLM(self.backbone.out_features, vocab_size, skip_id)

        torch_utils.ModuleManager.initialize_layers(self)

    def forward(self, x, segment_label, attention_mask=None, next_true=None, mask_true=None, **kwargs):
        x = self.backbone(x, segment_label, attention_mask)

        outputs = {}

        next_pred = None
        if self.is_nsp:
            next_pred = self.nsp(x)
            outputs['next_pred'] = next_pred

        mask_pred = None
        if self.is_mlm:
            mask_pred = self.mlm(x)
            outputs['mask_pred'] = mask_pred

        losses = {}
        if self.training:
            losses = self.loss(x.device, next_pred, mask_pred, next_true, mask_true)

        outputs.update(losses)
        return outputs

    def loss(self, device, next_pred=None, mask_pred=None, next_true=None, mask_true=None):
        losses = {}
        loss = torch.zeros(1, device=device)

        if self.is_nsp:
            next_loss = self.nsp.loss(next_pred, next_true)
            loss += next_loss
            losses['loss.next'] = next_loss

        if self.is_mlm:
            mask_loss = self.mlm.loss(mask_pred, mask_true)
            loss += mask_loss
            losses['loss.mask'] = mask_loss

        losses['loss'] = loss

        return losses


class NSP(nn.Module):
    """Next Sentence Prediction"""

    def __init__(self, in_features, out_features=2):
        super().__init__()
        self.fcn = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fcn(x[:, 0])  # select the first output
        return x

    def loss(self, next_pred, next_true):
        return F.cross_entropy(next_pred, next_true)


class MLM(nn.Sequential):
    """Masked Language Model"""

    def __init__(self, in_features, out_features, skip_id):
        self.skip_id = skip_id
        super().__init__(
            Linear(in_features, in_features, mode='lan', act=nn.GELU(), norm=nn.LayerNorm(in_features, eps=1e-12)),
            nn.Linear(in_features, out_features)
        )

    def loss(self, mask_pred, mask_true):
        mask_pred = mask_pred.transpose(1, 2)  # seq first -> class first
        return F.cross_entropy(mask_pred, mask_true, ignore_index=self.skip_id)


class Bert(nn.Module):
    def __init__(self, vocab_size, pad_id, max_seq_len=512, n_segment=2, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, drop_prob=0.1):
        super().__init__()
        self.embedding = EncoderEmbedding(vocab_size, hidden_size, pad_id, max_seq_len=max_seq_len, n_segment=n_segment, drop_prob=drop_prob)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, hidden_size * 4,
            drop_prob=drop_prob, num_blocks=num_hidden_layers
        )

        self.out_features = hidden_size

    def forward(self, x, segment_info, attention_mask=None):
        x = self.embedding(x, segment_info)
        x = self.encoder(x, attention_mask=attention_mask)
        return x
