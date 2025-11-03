from itertools import groupby

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import Paraformer
from .. import attentions


class Model(Paraformer.Model):
    """
    refer to:
        paper:
            (SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition)[https://arxiv.org/abs/2006.01713]
        code:
            `funasr.models.sense_voice.model.SenseVoiceSmall`
    """
    vocab_size: int = 25055
    use_ctc = True

    lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
    textnorm_dict = {"withitn": 14, "woitn": 15}

    def make_encoder(self, **encoder_config):
        self.embed = nn.Embedding(7 + len(self.lid_dict) + len(self.textnorm_dict), self.input_size)
        self.encoder = SenseVoiceEncoderSmall(input_size=self.input_size, **encoder_config)

    def make_decoder(self, **decoder_config):
        """no decoder"""

    def make_neck(self, **head_config):
        """no neck"""

    def encode(self, speech: torch.Tensor, speech_lens: torch.Tensor, use_itn=False, text_norm=None, language='auto', **kwargs):
        if text_norm is None:
            text_norm = "withitn" if use_itn else "woitn"

        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)

        event_emo_query = self.embed(
            torch.LongTensor([[1, 2]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)

        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[text_norm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)

        # specially add 4 token, [lang, event, emo, textnorm]
        speech = torch.cat((language_query, event_emo_query, textnorm_query, speech), dim=1)
        speech_lens += 4    # note, inplace mode
        attention_mask = attentions.make_pad_mask(speech_lens)[:, None, :].to(speech.device)
        return self.encoder(speech, speech_lens, attention_mask), attention_mask

    def post_process(self, encoder_out, encoder_out_mask, speech_lens, **kwargs):
        decoder_out = self.criterion_ctc.ctc_lo(encoder_out)
        ctc_logits = F.log_softmax(decoder_out, dim=2)

        preds = []
        sp_preds = []
        timestamps = []
        b = decoder_out.shape[0]
        for i in range(b):
            x = ctc_logits[i, : speech_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            timestamp = []
            token_ids = token_int[4:]

            if len(token_ids) == 0:
                continue

            logits_speech = F.softmax(decoder_out, dim=2)[i, 4: speech_lens[i].item(), :]
            pred = logits_speech.argmax(-1).cpu()
            logits_speech[pred == self.blank_id, self.blank_id] = 0
            align = self.ctc_forced_align(
                logits_speech.unsqueeze(0).float(),
                torch.Tensor(token_ids).unsqueeze(0).long().to(logits_speech.device),
                (speech_lens[i] - 4).long(),
                torch.tensor(len(token_ids)).unsqueeze(0).long().to(logits_speech.device),
            )
            pred = groupby(align[0, : speech_lens[i]])
            _start = 0
            token_id = 0
            ts_max = speech_lens[i] - 4
            for pred_token, pred_frame in pred:
                _end = _start + len(list(pred_frame))
                if pred_token != 0:
                    ts_left = max((_start * 60 - 30), 0)
                    ts_right = min((_end * 60 - 30), (ts_max * 60 - 30))
                    timestamp.append([ts_left, ts_right])
                    token_id += 1
                _start = _end

            sp_pred, pred = token_int[:4], token_int[4:]
            sp_preds.append(sp_pred)
            preds.append(pred)
            timestamps.append(timestamp)

        return dict(
            sp_preds=sp_preds,
            preds=preds,
            timestamps=timestamps,
        )

    def ctc_forced_align(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Align a CTC label sequence to an emission.
        Args:
            log_probs (Tensor): log probability of CTC emission output.
                Tensor of shape `(B, T, C)`. where `B` is the batch size, `T` is the input length,
                `C` is the number of characters in alphabet including blank.
            targets (Tensor): Target sequence. Tensor of shape `(B, L)`,
                where `L` is the target length.
            input_lengths (Tensor):
                Lengths of the inputs (max value must each be <= `T`). 1-D Tensor of shape `(B,)`.
            target_lengths (Tensor):
                Lengths of the targets. 1-D Tensor of shape `(B,)`.
        """
        targets[targets == self.ignore_id] = self.blank_id
        batch_size, input_time_size, _ = log_probs.size()
        bsz_indices = torch.arange(batch_size, device=input_lengths.device)
        _targets = torch.cat(
            (
                torch.stack((torch.full_like(targets, self.blank_id), targets), dim=-1).flatten(start_dim=1),
                torch.full_like(targets[:, :1], self.blank_id),
            ),
            dim=-1,
        )
        diff_labels = torch.cat(
            (
                torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
                _targets[:, 2:] != _targets[:, :-2],
            ),
            dim=1,
        )
        neg_inf = torch.tensor(float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
        padding_num = 2
        padded_t = padding_num + _targets.size(-1)
        best_score = torch.full((batch_size, padded_t), neg_inf, device=log_probs.device, dtype=log_probs.dtype)
        best_score[:, padding_num + 0] = log_probs[:, 0, self.blank_id]
        best_score[:, padding_num + 1] = log_probs[bsz_indices, 0, _targets[:, 1]]
        backpointers = torch.zeros((batch_size, input_time_size, padded_t), device=log_probs.device, dtype=targets.dtype)
        for t in range(1, input_time_size):
            prev = torch.stack(
                (best_score[:, 2:], best_score[:, 1:-1], torch.where(diff_labels, best_score[:, :-2], neg_inf))
            )
            prev_max_value, prev_max_idx = prev.max(dim=0)
            best_score[:, padding_num:] = log_probs[:, t].gather(-1, _targets) + prev_max_value
            backpointers[:, t, padding_num:] = prev_max_idx
        l1l2 = best_score.gather(
            -1, torch.stack((padding_num + target_lengths * 2 - 1, padding_num + target_lengths * 2), dim=-1)
        )
        path = torch.zeros((batch_size, input_time_size), device=best_score.device, dtype=torch.long)
        path[bsz_indices, input_lengths - 1] = padding_num + target_lengths * 2 - 1 + l1l2.argmax(dim=-1)
        for t in range(input_time_size - 1, 0, -1):
            target_indices = path[:, t]
            prev_max_idx = backpointers[bsz_indices, t, target_indices]
            path[:, t - 1] += target_indices - prev_max_idx
        alignments = _targets.gather(dim=-1, index=(path - padding_num).clamp(min=0))
        return alignments


class SenseVoiceEncoderSmall(Paraformer.SANMEncoder):
    tp_blocks = 20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tp_encoders = nn.ModuleList([self.make_sanm_encoder(self.output_size, self.output_size) for _ in range(self.tp_blocks)])

        self.tp_norm = Paraformer.LayerNorm(self.output_size)

    def forward(self, x, seq_lens, attention_mask):
        x = super().forward(x, seq_lens, attention_mask)

        for m in self.tp_encoders:
            x = m(x, attention_mask)

        x = self.tp_norm(x)

        return x
