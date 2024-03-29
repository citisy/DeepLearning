import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import bundles
from .transformers import TransformerBlock
from utils import torch_utils, math_utils


class Config(bundles.Config):
    default_model = '117M'

    @classmethod
    def make_full_config(cls):
        config_dict = {
            # https://openaipublic.blob.core.windows.net/gpt-2/models/117M/hparams.json
            '117M': {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "n_layer": 12
            },

            # https://openaipublic.blob.core.windows.net/gpt-2/models/345M/hparams.json
            '345M': {
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "n_layer": 24
            },

            # https://openaipublic.blob.core.windows.net/gpt-2/models/774M/hparams.json
            '774M': {
                "hidden_size": 1280,
                "num_attention_heads": 20,
                "n_layer": 36
            },

            # https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/hparams.json
            '1558M': {
                "hidden_size": 1600,
                "num_attention_heads": 25,
                "n_layer": 48
            }

        }

        # huggingface config
        config_dict.update(
            # https://huggingface.co/openai-community/gpt2
            small=config_dict['117M'],

            # https://huggingface.co/openai-community/gpt2-medium
            medium=config_dict['345M'],

            # https://huggingface.co/openai-community/gpt2-large
            large=config_dict['774M'],

            # https://huggingface.co/openai-community/gpt2-xl
            xl=config_dict['1558M'],
        )
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def from_openai_tf(cls, save_path, save_name='model.ckpt', n_layer=12):
        """model download from
        https://github.com/openai/gpt-2/blob/master/download_model.py"""
        info = [
            ('model/wte:0', None, None),
            ('model/wpe:0', None, None)
        ]

        for i in range(n_layer):
            tmp = [
                ('model/h%d/ln_1/g:0', 'w', 'n'),
                ('model/h%d/ln_1/b:0', 'b', 'n'),
                ('model/h%d/attn/c_attn/w:0', 'w', 'l'),
                ('model/h%d/attn/c_attn/b:0', 'b', 'l'),
                ('model/h%d/attn/c_proj/w:0', 'w', 'l'),
                ('model/h%d/attn/c_proj/b:0', 'b', 'l'),
                ('model/h%d/ln_2/g:0', 'w', 'n'),
                ('model/h%d/ln_2/b:0', 'b', 'n'),
                ('model/h%d/mlp/c_fc/w:0', 'w', 'l'),
                ('model/h%d/mlp/c_fc/b:0', 'b', 'l'),
                ('model/h%d/mlp/c_proj/w:0', 'w', 'l'),
                ('model/h%d/mlp/c_proj/b:0', 'b', 'l')
            ]
            tmp = [(t[0] % i, *t[1:]) for t in tmp]
            info += tmp

        info += [
            ('model/ln_f/g:0', 'w', 'n'),
            ('model/ln_f/b:0', 'b', 'n')
        ]

        var_name, key_types, value_types = math_utils.transpose(info)
        file_name = save_path
        state_dict = torch_utils.Load.from_tf_ckpt(file_name, var_names=var_name, key_types=key_types, value_types=value_types)

        return state_dict

    @classmethod
    def auto_download(cls, save_path, save_name=''):
        # download weight auto from transformers
        from transformers import GPT2PreTrainedModel

        model = GPT2PreTrainedModel.from_pretrained(save_path)
        state_dict = model.state_dict()
        return state_dict


class WeightConverter:
    @staticmethod
    def from_openai(state_dict):
        convert_dict = {
            'model.wte:0': 'embedding.token.weight',
            'model.wpe:0': 'embedding.position.weight',
            'model.h{0}.ln_1': 'encoder.{0}.res1.norm',
            'model.h{0}.attn.c_attn': 'encoder.{0}.res1.fn.to_qkv',
            'model.h{0}.attn.c_proj': 'encoder.{0}.res1.fn.to_out.linear',
            'model.h{0}.ln_2': 'encoder.{0}.res2.norm',
            'model.h{0}.mlp.c_fc': 'encoder.{0}.res2.fn.0.linear',
            'model.h{0}.mlp.c_proj': 'encoder.{0}.res2.fn.1.linear',
            'model.ln_f': 'norm',
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict

    @staticmethod
    def from_huggingface(state_dict):
        for k, v in state_dict.items():
            for a in ('c_attn', 'c_fc', 'c_proj'):
                if k.endswith(a + '.weight'):
                    state_dict[k] = v.T

        convert_dict = {
            'wte': 'embedding.token',
            'wpe': 'embedding.position',
            'h.{0}.ln_1': 'encoder.{0}.res1.norm',
            'h.{0}.attn.c_attn': 'encoder.{0}.res1.fn.to_qkv',
            'h.{0}.attn.c_proj': 'encoder.{0}.res1.fn.to_out.linear',
            'h.{0}.ln_2': 'encoder.{0}.res2.norm',
            'h.{0}.mlp.c_fc': 'encoder.{0}.res2.fn.0.linear',
            'h.{0}.mlp.c_proj': 'encoder.{0}.res2.fn.1.linear',
            'ln_f': 'norm',
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    """https://github.com/openai/gpt-2"""

    def __init__(self, vocab_size, max_seq_len=1024, hidden_size=768, num_attention_heads=12, n_layer=12,
                 drop_prob=0.1, pad_id=None):
        super().__init__()
        self.pad_id = pad_id
        self.n_layer = n_layer
        self.embedding = Embedding(vocab_size, hidden_size, pad_id, max_seq_len=max_seq_len)
        self.encoder = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, hidden_size * 4, norm_first=True, drop_prob=drop_prob,
                fn_kwargs=dict(separate=False)
            ) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.embedding_sim = EmbeddingSim(vocab_size, use_bias=False)

    def forward(self, x, **kwargs):
        if self.training:
            trues = torch.cat([x[:, 1:], torch.full((len(x), 1), self.pad_id)], dim=1)
            preds = self.decode(x)
            loss = self.loss(preds, trues)
            return {'loss': loss}
        else:
            return {'preds': self.post_process(x, **kwargs)}

    def loss(self, preds, trues):
        preds = preds.transpose(1, 2)  # seq first -> class first
        return F.cross_entropy(preds, trues)

    def post_process(self, x, seq_lens=None, max_gen_len=100, top_k=1):
        assert seq_lens is not None
        batch_size = len(x)
        for i in range(max_gen_len):
            output_data = self.decode(x)
            # add next preds
            x = torch.cat([x, torch.zeros((batch_size, 1)).to(x)], dim=-1)
            for index in range(batch_size):
                j = seq_lens[index] + i - 1
                preds = output_data[index, j]
                arg = torch.argsort(preds, descending=True)
                keep = arg[:top_k]
                preds = preds[keep]
                preds = preds / preds.sum()

                # random sampling
                next_id = keep[preds.multinomial(1)[0]]
                x[index][j + 1] = next_id
        return x

    def decode(self, sequence):
        x = self.embedding(sequence)
        mask = self.make_dynamic_mask(x)
        for component in self.encoder:
            x = component(x, attention_mask=mask)
        x = self.norm(x)
        preds = self.embedding_sim(x, self.embedding.token.weight)
        return preds

    def make_dynamic_mask(self, x):
        batch_size, seq_len, _ = x.size()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len).repeat(batch_size, 1, 1, 1)
        return mask


class Embedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding + SegmentEmbedding"""

    def __init__(self, vocab_size, embedding_dim, pad_id=None, max_seq_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_seq_len, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))
        self.embedding_dim = embedding_dim

    def forward(self, sequence):
        """(b, s) -> (b, s, h)
        note, s is a dynamic var"""
        x = (
                self.token(sequence)
                + self.position(self.position_ids[:, :sequence.shape[1]])
        )
        return x


class EmbeddingSim(nn.Module):
    def __init__(self, num_embeddings, use_bias=True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(num_embeddings))

    def forward(self, x, weight):
        y = x.matmul(weight.transpose(1, 0).detach())
        if self.use_bias:
            y += self.bias
        return F.softmax(y, dim=-1)
