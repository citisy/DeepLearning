import torch
import torch.nn.functional as F

from . import Paraformer
from .. import attentions


class Model(Paraformer.Model):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paper1: FunASR: A Fundamental End-to-End Speech Recognition Toolkit
    https://arxiv.org/abs/2305.11013
    Paper2: Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model
    https://arxiv.org/abs/2301.12343
    """

    def post_process(self, x, seq_lens=None, **kwargs):
        x, seq_lens = self.frontend(x, seq_lens)

        encoder_out, encoder_out_lens = self.encode(x, seq_lens)

        encoder_out_mask = attentions.make_pad_mask(encoder_out_lens, max_len=encoder_out.size(1))[:, None, :].to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, us_alphas, us_peaks = self.head(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)

        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []

        decoder_out = self.decode(encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length)

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            alphas = us_alphas[i][: encoder_out_lens[i] * 3]
            peaks = us_peaks[i][: encoder_out_lens[i] * 3]

            am_scores = decoder_out[i, : pre_token_length[i], :]
            token_int = am_scores.argmax(dim=-1)

            results.append({
                'alphas': alphas,
                'peaks': peaks,
                "token_int": token_int,
            })

        return results

    def decode(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
        decoder_out = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out = F.log_softmax(decoder_out, dim=-1)
        return decoder_out
