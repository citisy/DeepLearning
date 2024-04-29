import numpy as np
import torch
from torch import nn
from utils import torch_utils
from functools import partial


class PromptEncoder(nn.Module):
    def __init__(self, spell_length, hidden_size):
        super().__init__()
        self.spell_length = spell_length
        self.hidden_size = hidden_size

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

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self)

    def forward(self):
        seq_indices = torch.LongTensor(list(range(self.spell_length))).to(self.device)
        input_embeds = self.embedding(seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class ModelWarpForPT:
    def __init__(self, sp_id_dict, template, layer_name='token'):
        self.sp_id_dict = sp_id_dict
        self.layer_name = layer_name
        self.template = template
        self.cum_template = np.cumsum(template)
        self.spell_length = self.cum_template[-1]  # prompt length

    def warp(self, model):
        torch_utils.ModuleManager.freeze_module(model, allow_train=True)
        model.forward = partial(self.model_forward, model, model.forward)

        objs = torch_utils.ModuleManager.get_module_by_name(model, self.layer_name)
        assert len(objs), f'can not find embedding layer by input name {self.layer_name}'
        current_m, name, full_name = objs[0]
        emb_layer = getattr(current_m, name)
        prompt_encoder = PromptEncoder(self.spell_length, emb_layer.embedding_dim)
        prompt_encoder.to(torch_utils.ModuleInfo.possible_device(emb_layer))
        emb_layer.register_module('prompt_encoder', prompt_encoder)
        emb_layer.forward = partial(self.emb_forward, emb_layer, emb_layer.forward)

        return model

    def model_forward(
            self, base_layer, base_forward, x, *args,
            segment_label=None, attention_mask=None, token_cls_true=None,
            **kwargs
    ):
        x, flags = self.get_queries(x, x_t=token_cls_true)

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

        if token_cls_true is not None:
            tmp = []
            for i in range(x.shape[0]):
                a = torch.full_like(x[i], self.sp_id_dict['skip'])
                a[~flags[i]] = token_cls_true[i]
                tmp.append(a)
            token_cls_true = torch.stack(tmp)

        outputs = base_forward(
            x, *args,
            segment_label=segment_label, attention_mask=attention_mask, token_cls_true=token_cls_true,
            **kwargs
        )

        if 'token_cls_logit' in outputs:
            outputs['token_cls_logit'] = torch.stack(outputs['token_cls_logit'][~flags].chunk(x.shape[0]))

        return outputs

    def get_queries(self, x_h, **kwargs):
        raise NotImplementedError

    def emb_forward(self, base_layer, base_forward, x, *args, **kwargs):
        with torch.no_grad():
            raw_embeds = base_forward(x, *args, **kwargs)

        replace_embeds = base_layer.prompt_encoder()
        raw_embeds[x == self.sp_id_dict['prompt']] = replace_embeds.repeat(x.shape[0], 1).to(raw_embeds)

        return raw_embeds


class ModelWarpForBert(ModelWarpForPT):
    """
    Usages:
        .. code-block:: python

            # model having an embedding layer with name of 'token'
            model = ...
            model = ModelWarpForBert(sp_id_dict, layer_name='token').warp(model)
            model.to(device)

            # your train step
            ...
    """

    def __init__(self, sp_id_dict, template=(3, 3, 3), layer_name='token'):
        super().__init__(sp_id_dict, template, layer_name)

    def get_queries(self, x_h, **kwargs):
        """[C][X][S] -> [C][P][M][P][X][P][S]"""
        prompts = torch.full((self.spell_length,), self.sp_id_dict['prompt']).to(x_h)

        queries = []
        flags = []
        for x in x_h:
            idxes = (x == self.sp_id_dict['sep']).nonzero()
            if idxes.shape[0] > 0:
                idx = idxes[-1]
            else:
                idx = len(x)

            q = [
                x[0:1],
                prompts[0: self.cum_template[0]],
                torch.full((1,), self.sp_id_dict['mask']).to(x_h),
                prompts[self.cum_template[0]: self.cum_template[1]],
                x[1:idx],
                prompts[self.cum_template[1]: self.cum_template[2]],
                x[idx:]
            ]
            q = torch.cat(q)
            flag = torch.zeros((len(q),), device=q.device, dtype=torch.bool)
            flag[1:1 + 1 + self.cum_template[1]] = True
            flag[1 + self.cum_template[1] + idx: 1 + self.cum_template[2] + idx] = True

            queries.append(q)
            flags.append(flag)

        queries = torch.stack(queries)
        flags = torch.stack(flags)
        return queries, flags


class ModelWarpForGpt(ModelWarpForPT):
    """
    Usages:
        .. code-block:: python

            # model having an embedding layer with name of 'token'
            model = ...
            model = ModelWarpForGpt(sp_id_dict, layer_name='token').warp(model)
            model.to(device)

            # your train step
            ...
    """

    def __init__(self, sp_id_dict, template=(3, 3), layer_name='token'):
        super().__init__(sp_id_dict, template, layer_name)

    def get_queries(self, x_h, x_t=None, **kwargs):
        """
        [xh][xt] -> [P][xh][p][xt]
        """
        b, s = x_h.shape
        prompts = torch.full((self.spell_length,), self.sp_id_dict['prompt']).to(x_h)

        queries = []
        flags = []
        for i in range(b):
            q = torch.cat([
                prompts[0: self.cum_template[0]],
                torch.full((1,), self.sp_id_dict['pad']).to(x_h),  # add token of ' '
                x_h[i],
                prompts[self.cum_template[0]: self.cum_template[1]],
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

            flag = torch.zeros((q.shape[1], ), device=q.device, dtype=torch.bool)
            flag[:self.cum_template[0] + 1] = True
            flag[self.cum_template[0] + 1 + len(x_h[i]): self.cum_template[1] + 1 + len(x_h[i])] = True
            if x_t is not None:
                flag[self.cum_template[1] + 1 + len(x_h[i])] = True
            flag[-1] = True

            queries.append(q)
            flags.append(flag)

        queries = torch.stack(queries)
        flags = torch.stack(flags)
        return queries, flags
