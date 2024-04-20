import numpy as np
import torch
from torch import nn
from utils import torch_utils
from functools import partial


class PromptEncoder(nn.Module):
    def __init__(self, spell_length, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.seq_indices = torch.LongTensor(list(range(spell_length)))
        self.embedding = nn.Embedding(spell_length, self.hidden_size)
        self.lstm_head = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )  # Bi-LSTM
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    @property
    def weight(self):
        """for inference"""
        return self.forward()

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class ModelWarpForPT:
    def __init__(self, sp_id_dict, template, embedding_layer_name='token'):
        self.sp_id_dict = sp_id_dict
        self.embedding_layer_name = embedding_layer_name
        self.template = template
        self.cum_template = np.cumsum(template)
        self.spell_length = self.cum_template[-1]  # prompt length

    def warp(self, model):
        model.forward = partial(self.model_forward, model.forward)

        objs = torch_utils.ModuleManager.get_module_by_name(model, self.embedding_layer_name)
        assert len(objs), f'can not find embedding layer by input name {self.embedding_layer_name}'
        current_m, name, full_name = objs[0]
        emb_layer = getattr(current_m, name)
        self.prompt_encoder = PromptEncoder(self.spell_length, emb_layer.embedding_dim)
        emb_layer.forward = partial(self.emb_forward, emb_layer.forward)

        return model

    def model_forward(
            self, forward_func, x, *args,
            segment_label=None, attention_mask=None, x_t=None,
            **kwargs
    ):
        x = self.get_queries(x, x_t=x_t)

        if segment_label is not None:
            segment_label = torch.cat((
                torch.zeros((x.shape[0], self.spell_length + 1)).to(segment_label),
                segment_label
            ), dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat((
                torch.ones((x.shape[0], self.spell_length + 1)).to(attention_mask),
                attention_mask
            ), dim=1)

        return forward_func(
            x, *args,
            segment_label=segment_label, attention_mask=attention_mask,
            **kwargs
        )

    def get_queries(self, x_h, **kwargs):
        raise NotImplementedError

    def emb_forward(self, forward_func, x, *args, **kwargs):
        with torch.no_grad():
            raw_embeds = forward_func(x, *args, **kwargs)

        replace_embeds = self.prompt_encoder()
        raw_embeds[x == self.sp_id_dict['prompt']] = replace_embeds.repeat(x.shape[0], 1).to(raw_embeds)

        return raw_embeds


class ModelWarpForBert(ModelWarpForPT):
    """
    Usages:
        .. code-block:: python

            model = ...
            model = ModelWarpForBert(sp_id_dict).warp(model)

            # your train step
            ...
    """
    def __init__(self, sp_id_dict, template=(3, 3, 3), embedding_layer_name='token'):
        super().__init__(sp_id_dict, template, embedding_layer_name)

    def get_queries(self, x_h, **kwargs):
        """[C][X][S] -> [C][P][M][P][X][P][S]"""
        prompts = torch.full((self.spell_length,), self.sp_id_dict['prompt']).to(x_h)

        queries = []
        for x in x_h:
            idx = (x == self.sp_id_dict['sep']).nonzero()[0]
            q = [
                x[0:1],
                prompts[0: self.cum_template[0]],
                torch.full((1,), self.sp_id_dict['mask']).to(x_h),
                prompts[self.cum_template[0]: self.cum_template[1]],
                x[1:idx],
                prompts[self.cum_template[1]: self.cum_template[2]],
                x[idx:]
            ]

            queries.append(torch.cat(q))

        queries = torch.stack(queries)
        return queries


class ModelWarpForGpt(ModelWarpForPT):
    """
    Usages:
        .. code-block:: python

            model = ...
            model = ModelWarpForGpt(sp_id_dict).warp(model)

            # your train step
            ...
    """

    def __init__(self, sp_id_dict, template=(3, 3), embedding_layer_name='token'):
        super().__init__(sp_id_dict, template, embedding_layer_name)

    def get_queries(self, x_h, x_t=None, **kwargs):
        """
        [xh][xt] -> [P][xh][p][xt]
        """
        b, s = x_h.shape
        prompts = torch.full((self.spell_length,), self.sp_id_dict['prompt']).to(x_h)

        queries = []
        for i in range(b):
            q = torch.cat([
                prompts[0: self.template[0]],
                torch.full((1,), self.sp_id_dict['pad']).to(x_h),  # add token of ' '
                x_h[i],
                prompts[self.template[0]: self.template[1]],
            ])
            c = 1

            if x_t is not None:
                q = torch.cat([
                    q,
                    torch.full((1,), self.sp_id_dict['pad']).to(x_h),  # add token of ' '
                    x_t[i],
                ])
                c += 1

            q = torch.cat([
                q,
                torch.full((b - self.spell_length - c,), self.sp_id_dict['pad']).to(x_h),  # add token of ' '
            ])

            queries.append(q)

        queries = torch.stack(queries)
        return queries
