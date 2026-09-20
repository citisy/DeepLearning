import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from ..image_classification import PPLCNetV4
from utils import math_utils, torch_utils, converter
from . import BaseTextRecModel
from .. import activations, bundles
from ..layers import Conv
from ..text_pretrain.transformers import TransformerSequential
from . import PPOCRv4_rec


class Config(bundles.Config):
    tiny_neck = dict(
        encoder_type='reshape',
    )
    small_neck = dict(
        encoder_type='lightsvtr',
        depth=2,
        out_ch=120,
        mlp_ratio=2.0,
        local_kernel=7,
    )
    medium_neck = dict(
        encoder_type='lightsvtr',
        depth=2,
        out_ch=192,
        mlp_ratio=4.0,
        local_kernel=7,
    )

    tiny_head = dict(
        out_ch=6906,
        mid_ch=80,
        use_guide=True,
    )
    small_head = dict(
        out_ch=18710
    )
    medium_head = dict(
        out_ch=18710
    )

    tiny_loss = dict(
        nrtr_dim=384,
        max_text_length=25,
    )
    small_loss = dict(
        nrtr_dim=384,
        max_text_length=25,
    )
    medium_loss = dict(
        nrtr_dim=512,
        max_text_length=25,
    )

    default_model = 'medium'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'tiny': dict(
                backbone_config=PPLCNetV4.Config.rec_tiny_backbone,
                neck_config=cls.tiny_neck,
                head_config=cls.tiny_head,
                loss_config=cls.tiny_loss
            ),
            'small': dict(
                backbone_config=PPLCNetV4.Config.rec_small_backbone,
                neck_config=cls.small_neck,
                head_config=cls.small_head,
                loss_config=cls.small_loss
            ),
            'medium': dict(
                backbone_config=PPLCNetV4.Config.rec_medium_backbone,
                neck_config=cls.medium_neck,
                head_config=cls.medium_head,
                loss_config=cls.medium_loss
            )
        }


class WeightConverter(PPOCRv4_rec.WeightConverter):
    neck_convert_dict = {
        'head.ctc_encoder': 'neck',
        'head.ctc_encoder.encoder.svtr_block.{0}.norm1': 'neck.encoder.svtr_block.{0}.attn_res.norm',
        'head.ctc_encoder.encoder.svtr_block.{0}.mixer.qkv': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_qkv',
        'head.ctc_encoder.encoder.svtr_block.{0}.mixer.proj': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_out.linear',
        'head.ctc_encoder.encoder.svtr_block.{0}.norm2': 'neck.encoder.svtr_block.{0}.ff_res.norm',
        'head.ctc_encoder.encoder.svtr_block.{0}.mlp.fc1': 'neck.encoder.svtr_block.{0}.ff_res.fn.0.linear',
        'head.ctc_encoder.encoder.svtr_block.{0}.mlp.fc2': 'neck.encoder.svtr_block.{0}.ff_res.fn.1.linear',
        'head.ctc_encoder.encoder.local_conv.0': 'neck.encoder.local_conv.conv',
        'head.ctc_encoder.encoder.local_conv.1': 'neck.encoder.local_conv.norm',
    }

    head_convert_dict = {
        'head.ctc_head': 'head',
        'head.ctc_head.fc1': 'head.fc.0',
        'head.ctc_head.fc2': 'head.fc.1',
        'head.before_gtc': 'criterion.before_gtc',
        'head.gtc_head': 'criterion.gtc_head',
    }

    @classmethod
    def from_paddle(cls, state_dict):
        """
        tiny: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_rec_pretrained.pdparams
        small: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
        medium: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
        """
        state_dict = cls.pre_convert(state_dict)
        convert_dict = {
            **PPLCNetV4.WeightConverter.backbone_convert_dict,
            **cls.neck_convert_dict,
            **cls.head_convert_dict
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict

    transformers_neck_convert_dict = {
        'head.encoder.conv_block.0.convolution': 'neck.encoder.conv_reduce.conv',
        'head.encoder.conv_block.0.normalization': 'neck.encoder.conv_reduce.norm',
        'head.encoder.conv_block.1.convolution': 'neck.encoder.skip_conv.conv',
        'head.encoder.conv_block.1.normalization': 'neck.encoder.skip_conv.norm',
        'head.encoder.conv_block.2.convolution': 'neck.encoder.local_conv.conv',
        'head.encoder.conv_block.2.normalization': 'neck.encoder.local_conv.norm',
        'head.encoder.norm': 'neck.encoder.norm',

        'head.encoder.svtr_block.{0}.layer_norm1': 'neck.encoder.svtr_block.{0}.attn_res.norm',
        'head.encoder.svtr_block.{0}.layer_norm2': 'neck.encoder.svtr_block.{0}.ff_res.norm',
        'head.encoder.svtr_block.{0}.mlp.fc1': 'neck.encoder.svtr_block.{0}.ff_res.fn.0.linear',
        'head.encoder.svtr_block.{0}.mlp.fc2': 'neck.encoder.svtr_block.{0}.ff_res.fn.1.linear',
        'head.encoder.svtr_block.{0}.self_attn.projection': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_out.linear',
        'head.encoder.svtr_block.{0}.self_attn.qkv': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_qkv',
    }

    transformers_head_convert_dict = {
        'head.head': 'head.fc'
    }

    @classmethod
    def from_transformers(cls, state_dict):
        """
        tiny: https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv6_tiny_rec_safetensors
        small: https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv6_small_rec_safetensors
        medium: https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv6_medium_rec_safetensors
        """
        convert_dict = {
            **PPLCNetV4.WeightConverter.transformers_backbone_convert_dict,
            **cls.transformers_neck_convert_dict,
            **cls.transformers_head_convert_dict
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(BaseTextRecModel):
    def __init__(
            self,
            backbone_config=PPLCNetV4.Config.rec_medium_backbone,
            neck_config=Config.medium_neck,
            head_config=Config.medium_head,
            loss_config=Config.medium_loss,
            **kwargs
    ):
        backbone = PPLCNetV4.Backbone(**backbone_config)
        neck = SequenceEncoder(in_ch=backbone.out_channels, **neck_config)
        head = PPOCRv4_rec.CTCHead(in_ch=neck.out_channels, **head_config)
        super().__init__(
            out_features=head.out_channels,
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )
        self.criterion = Loss(backbone.out_channels, head.out_channels, **loss_config)

    def set_inference_only(self):
        del self.criterion

    def post_process(self, x):
        probs, preds = torch.max(x, -1)
        words = []
        for b in range(x.shape[0]):
            pred = preds[b]
            diff = torch.diff(pred)
            diff = torch.cat([torch.tensor([-1]).to(diff), diff])
            pred = pred[diff != 0]
            pred = pred[pred != 0]
            chars = [self.id2char[int(i)] for i in pred]
            words.append(''.join(chars))

        return {'preds': words}

    def loss(self, probs, true_label):
        gtc = self.criterion(probs, true_label)
        raise NotImplementedError


class Model4Export(Model):
    """for exporting to onnx, torchscript, etc."""

    mean = 127.5
    std = 127.5

    max_h = 48
    max_w = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # torch.jit.trace bakes arange(...).to(x.device) as cuda:0; buffers follow jit.load / Triton device.
        self.register_buffer('h_indices', torch.arange(self.max_h), persistent=False)
        self.register_buffer('w_indices', torch.arange(self.max_w), persistent=False)

    def forward(self, x, ds, rs):
        x = self.pre_process(x, ds, rs)
        x = self.process(x)
        x = x.to(torch.float16)
        return x

    def pre_process(self, x, ds, rs):
        """for faster infer, use uint8 input and fp32 to output"""
        x = x.to(torch.float32)
        x = (x - self.mean) / self.std
        h_mask = self.h_indices >= (self.max_h - ds)
        x = x.masked_fill(h_mask[:, None, :, None], 0)
        w_mask = self.w_indices >= (self.max_w - rs)
        x = x.masked_fill(w_mask[:, None, None, :], 0)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_ch, encoder_type='lightsvtr', **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_reshape = Rearrange('b c 1 w -> b w c')
        if encoder_type in ('svtr', 'lightsvtr'):
            self.encoder = EncoderWithLightSVTR(in_ch, **kwargs)
            self.out_channels = self.encoder.out_channels
        else:
            self.out_channels = in_ch

    def forward(self, x):
        if self.encoder_type in ('svtr', 'lightsvtr'):
            x = self.encoder(x)
        x = self.encoder_reshape(x)
        return x


class EncoderWithLightSVTR(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch=64,
            depth=1,
            num_heads=8,
            mlp_ratio=4.0,
            drop_prob=0.1,
            local_kernel=7,
            **kwargs
    ):
        super().__init__()
        self.conv_reduce = Conv(in_ch, out_ch, 1, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=activations.Swish())
        self.local_conv = Conv(out_ch, out_ch, [1, local_kernel], bias=False, groups=out_ch, mode='cna', norm_fn=nn.BatchNorm2d, act=activations.Swish())

        self.svtr_block = TransformerSequential(
            out_ch,
            num_heads,
            int(out_ch * mlp_ratio),
            drop_prob=drop_prob,
            norm_first=True,
            fn_kwargs=dict(
                separate=False
            ),
            norm_kwargs=dict(
                eps=1e-05
            ),
            num_blocks=depth
        )
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)
        self.skip_conv = Conv(in_ch, out_ch, 1, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=activations.Swish())
        self.out_channels = out_ch

    def forward(self, x):
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        z = z + self.local_conv(z)
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)

        z = self.svtr_block(z)

        z = self.norm(z)
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = z + skip
        return z


class Loss(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            nrtr_dim=256, max_text_length=25, num_decoder_layers=4,
            **kwargs
    ):
        super().__init__()
        self.before_gtc = nn.Sequential(
            nn.Flatten(2),
            FCTranspose(in_ch, nrtr_dim)
        )
        self.gtc_head = NRTRHead(
            d_model=nrtr_dim,
            nhead=nrtr_dim // 32,
            num_decoder_layers=num_decoder_layers,
            max_len=max_text_length,
            dim_feedforward=nrtr_dim * 4,
            out_channels=out_ch + 3,
        )

    def forward(self, x, true_label):
        gtc = self.gtc_head(self.before_gtc(x), true_label)
        return gtc


class FCTranspose(nn.Module):
    def __init__(self, in_ch, out_ch, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not only_transpose:
            self.fc = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.only_transpose:
            return x
        return self.fc(x)


class NRTRHead(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_decoder_layers=4,
            max_len=25,
            dim_feedforward=1024,
            attention_dropout_rate=0.0,
            residual_dropout_rate=0.1,
            out_channels=0,
            scale_embedding=True,
            **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(dropout=residual_dropout_rate, dim=d_model)
        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model,
                nhead,
                dim_feedforward,
                attention_dropout_rate,
                residual_dropout_rate,
                with_self_attn=True,
                with_cross_attn=True,
            )
            for _ in range(num_decoder_layers)
        ])
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels, bias=False)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1], tgt.device)
        memory = src
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        return self.tgt_word_prj(tgt)

    def forward_test(self, src):
        bs = src.shape[0]
        memory = src
        dec_seq = torch.full((bs, 1), 2, dtype=torch.long, device=src.device)
        dec_prob = torch.full((bs, 1), 1.0, dtype=src.dtype, device=src.device)
        for _ in range(1, self.max_len):
            dec_seq_embed = self.positional_encoding(self.embedding(dec_seq))
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1], src.device)
            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            word_prob = F.softmax(self.tgt_word_prj(tgt[:, -1, :]), dim=-1)
            preds_idx = torch.argmax(word_prob, dim=-1)
            if torch.equal(preds_idx, torch.full_like(preds_idx, 3)):
                break
            preds_prob = torch.max(word_prob, dim=-1).values
            dec_seq = torch.cat([dec_seq, preds_idx.reshape(-1, 1)], dim=1)
            dec_prob = torch.cat([dec_prob, preds_prob.reshape(-1, 1)], dim=1)
        return [dec_seq, dec_prob]

    def forward(self, src, targets=None):
        if self.training and targets is not None:
            max_len = targets[1].max()
            tgt = targets[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        return self.forward_test(src)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        b, qn, _ = query.shape
        if self.self_attn:
            qkv = self.qkv(query).reshape(b, qn, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kn = key.shape[1]
            q = self.q(query).reshape(b, qn, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(key).reshape(b, kn, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, qn, self.embed_dim)
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            attention_dropout_rate=0.0,
            residual_dropout_rate=0.1,
            with_self_attn=True,
            with_cross_attn=False,
            epsilon=1e-5,
    ):
        super().__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=True)
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout_rate, self_attn=False)
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.mlp = NRTRMlp(d_model, dim_feedforward, drop_prob=residual_dropout_rate)
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt = self.norm1(tgt + self.dropout1(self.self_attn(tgt, attn_mask=self_mask)))
        if self.with_cross_attn:
            tgt = self.norm2(tgt + self.dropout2(self.cross_attn(tgt, key=memory, attn_mask=cross_mask)))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class NRTRMlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x).transpose(0, 1)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        x = self.embedding(x)
        if self.scale_embedding:
            x = x * math.sqrt(self.d_model)
        return x
