import torch
import torch.nn.functional as F
from torch import nn

from utils import torch_utils
from . import XLMRoberta, bert
from ..layers import Linear


class WeightConverter:
    @classmethod
    def from_hf(cls, state_dicts):
        state_dict = {}
        for k, v in state_dicts.items():
            if k == 'backbone':
                convert_dict = XLMRoberta.WeightConverter.convert_dict
            elif k == 'sparse':
                convert_dict = {
                    'weight': 'sparse_head.fcn.linear.weight',
                    'bias': 'sparse_head.fcn.linear.bias',
                }
            elif k == 'colbert':
                convert_dict = {
                    'weight': 'colbert_head.fcn.linear.weight',
                    'bias': 'colbert_head.fcn.linear.bias',
                }
            else:
                raise

            state_dict.update(torch_utils.Converter.convert_keys(v, convert_dict))

        return state_dict


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = bert.Bert(**XLMRoberta.Config.backbone)
        self.dense_head = DenseHead()
        self.sparse_head = SparseHead(self.backbone.out_features, 1)
        self.colbert_head = ColbertHead(self.backbone.out_features, self.backbone.out_features)

        self.step = 0

    def forward(self, *args, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.inference(*args, **kwargs)

    def inference(
            self, text_ids, attention_mask,
            return_dense=True,
            return_sparse=False, return_sparse_embedding=False,
            return_colbert=False,
    ):
        h = self.backbone(text_ids, attention_mask=attention_mask)

        output = {}
        if return_dense:
            output['dense_vecs'] = self.dense_head(h, attention_mask)
        if return_sparse:
            output['sparse_vecs'] = self.sparse_head(h, text_ids, return_embedding=return_sparse_embedding)
        if return_colbert:
            output['colbert_vecs'] = self.colbert_head(h, attention_mask)

        return output


class DenseHead(nn.Module):
    def forward(self, hidden_state, attention_mask, sentence_pooling_method='cls'):
        """Use the pooling method to get the dense embedding."""
        if sentence_pooling_method == "cls":
            embedding = hidden_state[:, 0]
        elif sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        elif sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                embedding = hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                embedding = hidden_state[
                    torch.arange(batch_size, device=hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {sentence_pooling_method} not implemented")

        embedding = F.normalize(embedding, dim=-1)
        return embedding


class SparseHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fcn = Linear(in_features, out_features, mode='la', act=nn.ReLU())

    def forward(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = self.fcn(hidden_state)
        if not return_embedding:
            return token_weights[:, :, 0]

        if self.training:
            embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size).to(token_weights)
            embedding = torch.scatter(embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)
            embedding = torch.max(embedding, dim=1).values
        else:
            # Optimize suggestion from issue #1364: https://github.com/FlagOpen/FlagEmbedding/issues/1364
            # Disable when self.training = True, otherwise will cause:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            embedding = torch.zeros(input_ids.size(0), self.vocab_size).to(token_weights)
            embedding = embedding.scatter_reduce(dim=-1, index=input_ids, src=token_weights.squeeze(-1), reduce="amax")

        unused_tokens = [self.cls_id, self.eos_id,self.pad_id, self.unk_tid]
        embedding[:, unused_tokens] *= 0.
        return embedding


class ColbertHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fcn = Linear(in_features, out_features, mode='l')

    def forward(self, hidden_state, attention_mask):
        embedding = self.fcn(hidden_state[:, 1:])
        embedding = embedding * attention_mask[:, 1:][:, :, None].float()
        embedding = F.normalize(embedding, dim=-1)
        return embedding
