import math
import torch
from torch import nn
import torch.nn.functional as F
from utils import torch_utils
from ..layers import Linear
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

tiny = dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=2)
mini = dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4)
small = dict(hidden_size=512, num_hidden_layers=4, num_attention_heads=8)
medium = dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8)
base = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
large = dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16)


class Model(nn.Module):
    """refer to
    paper:
        - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    code:
        - https://github.com/google-research/bert
        - https://github.com/codertimo/BERT-pytorch
    """

    def __init__(self, vocab_size, sp_tag_dict, bert_config=base):
        super().__init__()
        self.sp_tag_dict = sp_tag_dict

        self.bert = Bert(vocab_size, sp_tag_dict, **bert_config)
        self.nsp = NSP(self.bert.hidden_size)
        self.mlm = MLM(self.bert.hidden_size, vocab_size, sp_tag_dict['non_mask'])

        torch_utils.initialize_layers(self)

    def forward(self, x, segment_label, next_true=None, mask_true=None):
        x = self.bert(x, segment_label)
        next_pred = self.nsp(x)
        mask_pred = self.mlm(x)

        losses = {}
        if self.training:
            losses = self.loss(next_pred, mask_pred, next_true, mask_true)

        return dict(
            next_pred=next_pred,
            mask_pred=mask_pred,
            **losses
        )

    def loss(self, next_pred, mask_pred, next_true=None, mask_true=None):
        next_loss = self.nsp.loss(next_pred, next_true)
        mask_loss = self.mlm.loss(mask_pred, mask_true)

        return {
            'loss.next': next_loss,
            'loss.mask': mask_loss,
            'loss': next_loss + mask_loss
        }

    def post_process(self):
        pass


class NSP(nn.Module):
    """Next Sentence Prediction"""

    def __init__(self, input_size, output_size=2):
        super().__init__()
        self.fcn = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fcn(x[:, 0])  # select the first output
        return x

    def loss(self, next_pred, next_true):
        return F.cross_entropy(next_pred, next_true)


class MLM(nn.Module):
    """Masked Language Model"""

    def __init__(self, input_size, output_size, non_mask_tag):
        super().__init__()
        self.non_mask_tag = non_mask_tag
        self.fcn = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fcn(x)
        x = x.transpose(1, 2)
        return x

    def loss(self, mask_pred, mask_true):
        return F.cross_entropy(mask_pred, mask_true, ignore_index=self.non_mask_tag)


class Bert(nn.Module):
    def __init__(self, vocab_size, sp_tag_dict, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, drop_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sp_tag_dict = sp_tag_dict

        self.embedding = BERTEmbedding(vocab_size, hidden_size, sp_tag_dict)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_size, num_attention_heads, hidden_size * 4, drop_prob) for _ in range(num_hidden_layers)])

    def forward(self, x, segment_info):
        mask = (x == self.sp_tag_dict['pad'])[:, None, None].repeat(1, 1, x.size(1), 1)  # mask pad
        x = self.embedding(x, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class BERTEmbedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding + SegmentEmbedding"""

    def __init__(self, vocab_size, embedding_dim, sp_tag_dict, drop_prob=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=sp_tag_dict['pad'])
        self.position = PositionalEmbedding(self.token.embedding_dim)
        self.segment = nn.Embedding(3, self.token.embedding_dim, padding_idx=sp_tag_dict['seg_pad'])
        self.dropout = nn.Dropout(drop_prob)
        self.embedding_dim = embedding_dim

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """MultiHeadedAttention + PositionWiseFeedForward"""

    def __init__(self, hidden_size, num_attention_heads, feed_forward_hidden, drop_prob=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(num_attention_heads, d_model=hidden_size)
        self.feed_forward = PositionWiseFeedForward(hidden_size, feed_forward_hidden, drop_prob=drop_prob)
        self.res1 = Residual(hidden_size, drop_prob=drop_prob)
        self.res2 = Residual(hidden_size, drop_prob=drop_prob)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        x = self.res1(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.res2(x, self.feed_forward)
        return self.dropout(x)


class Residual(nn.Module):
    def __init__(self, input_size, drop_prob=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size, d_model, dropout=0.1):
        super().__init__()
        assert d_model % hidden_size == 0

        d_k = d_model // hidden_size
        self.to_qkv = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model),
            Rearrange('b s (h dk)-> b h s dk', dk=d_k, h=hidden_size)
        ) for _ in range(3)])
        self.to_out = nn.Sequential(
            Rearrange('b h s dk -> b s (h dk)'),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def attend(self, q, k, v, mask=None):
        """Scaled Dot-Product Attention
        attn(q, k, v) = softmax(qk'/sqrt(dk))*v"""
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:  # mask pad
            s = s.masked_fill(mask, -1e9)

        attn = F.softmax(s, dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)

        return attn

    def forward(self, q, k, v, mask=None):
        q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        x = self.attend(q, k, v, mask=mask)

        return self.to_out(x)


class PositionWiseFeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super().__init__(
            Linear(d_model, d_ff, mode='lad', act=nn.GELU(), drop_prob=drop_prob),
            Linear(d_ff, d_model, mode='l')
        )
