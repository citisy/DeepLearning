import torch
from torch import nn
from torch.nn import functional as F

from utils import torch_utils


class ModelWrap:
    """
    References:
        paper:
            - [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

    Usages:
        .. code-block:: python

            model = Model()
            ref_model = copy.deepcopy(model)
            model_wrap = ModelWrap(ref_model, police_func, ref_func)
            model_wrap.wrap(model)

            # define your train step
            opti = ...
            model(data)
            ...
    """

    def __init__(self, ref_model, police_func, ref_func, beta=0.1):
        self.criterion = DpoLoss()
        torch_utils.ModuleManager.freeze_module(ref_model)
        self.ref_model = ref_model
        self.police_func = police_func
        self.ref_func = ref_func
        self.model = None
        self.beta = beta

    def wrap(self, model: nn.Module):
        def fit(text_ids, label_ids, *args, attention_mask=None, chosen_idx=None, **kwargs):
            attention_mask = attention_mask.type(torch.long)
            with torch.no_grad():
                ref_logits = self.ref_func(text_ids, **kwargs)
                ref_logps = self.logits_to_logps(ref_logits, label_ids, attention_mask)
                ref_chosen_logps = ref_logps[chosen_idx]
                ref_rejected_logps = ref_logps[~chosen_idx]

            logits = self.police_func(text_ids, **kwargs)
            logps = self.logits_to_logps(logits, label_ids, attention_mask)
            policy_chosen_logps = logps[chosen_idx]
            policy_rejected_logps = logps[~chosen_idx]
            return self.criterion(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=self.beta)

        model.fit = fit
        self.model = model
        return model

    def logits_to_logps(self, logits, labels, attention_mask):
        log_probs = F.log_softmax(logits, dim=2)
        logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logps = (logps * attention_mask).sum(dim=1) / seq_lengths.squeeze()
        return logps


class DpoLoss(nn.Module):
    def forward(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

        # only for logging
        loss_chosen = beta * (policy_chosen_logps - ref_chosen_logps).detach().mean()
        loss_rejected = beta * (policy_rejected_logps - ref_rejected_logps).detach().mean()

        return {
            'loss': loss,
            'loss.chosen': loss_chosen,
            'loss.rejected': loss_rejected,
        }
