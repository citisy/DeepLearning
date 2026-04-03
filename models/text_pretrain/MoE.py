import torch
from torch import nn
import torch.nn.functional as F

from . import llama, transformers


@transformers.make_ff_fn.add_register()
class MoeFeedForward(nn.Module):
    def __init__(
            self,
            hidden_size, ff_hidden_size,
            experts_hidden_size=1408, share_experts_hidden_size=5632,
            num_experts=60, top_k=4, norm_topk_prob=False,
            **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # gating
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [llama.FeedForward(hidden_size, experts_hidden_size, **kwargs) for _ in range(num_experts)]
        )

        self.shared_expert = llama.FeedForward(hidden_size, share_experts_hidden_size, **kwargs)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states, ff_cache=None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing, so we'll use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        if ff_cache is not None:
            ff_cache['router_logits'] = router_logits
        return final_hidden_states


class MoeLoss(nn.Module):
    def __init__(self, num_experts=60, top_k=4, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, ff_caches, mask=None):
        """

        Args:
            ff_caches (dict): come from `MoeFeedForward.forward`
            mask (torch.Tensor): shape of (b, s), came from `attentions.make_pad_mask`

        """
        router_logits = [ff_cache['router_logits'] for ff_cache in ff_caches]
        concatenated_gate_logits = torch.cat(router_logits, dim=0)

        routing_weights = F.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        expert_mask = F.one_hot(selected_experts, self.num_experts)

        if mask is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
        else:
            batch_size, sequence_length = mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, self.top_k, self.num_experts))
                .reshape(-1, self.top_k, self.num_experts)
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, routing_weights.shape[1]))
                .reshape(-1, routing_weights.shape[1])
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss * self.num_experts
