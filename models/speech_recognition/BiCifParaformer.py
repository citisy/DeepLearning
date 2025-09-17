import torch
import torch.nn.functional as F

from . import Paraformer
from .Paraformer import WeightConverter
from ..bundles import WeightLoader
from .. import attentions


class Model(Paraformer.Model):
    """refer to:
    paper:
        - [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
        - [Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model](https://arxiv.org/abs/2301.12343)
    """

    def fit(self, speech, speech_lens, **kwargs):
        encoder_out, encoder_out_mask = self.process(speech, speech_lens)
        return self.loss(encoder_out, encoder_out_mask, speech_lens, **kwargs)

    def inference(self, speech, speech_lens, **kwargs):
        encoder_out, encoder_out_mask = self.process(speech, speech_lens)
        return self.post_process(encoder_out, encoder_out_mask, speech_lens, **kwargs)

    def process(self, speech, speech_lens):
        encoder_out = self.encode(speech, speech_lens)
        encoder_out_mask = attentions.make_pad_mask(speech_lens, max_len=encoder_out.shape[1])[:, None, :].to(encoder_out.device)
        return encoder_out, encoder_out_mask

    def post_process(self, encoder_out, encoder_out_mask, speech_lens, **kwargs):
        # predict the length of token
        pre_acoustic_embeds, token_num, alphas, us_alphas, us_peaks = self.neck(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)

        token_num = token_num.round().long()
        if torch.max(token_num) < 1:
            return {}

        decoder_out = self.decode(encoder_out, speech_lens, pre_acoustic_embeds, token_num)

        preds = []
        timestamps = []
        b, n, d = decoder_out.size()
        for i in range(b):
            alphas = us_alphas[i][: speech_lens[i] * self.neck.upsample_times]
            peaks = us_peaks[i][: speech_lens[i] * self.neck.upsample_times]

            am_scores = decoder_out[i, : token_num[i], :]
            pred = am_scores.argmax(dim=-1)
            keep = (pred != self.blank_id) & (pred != self.sos_id) & (pred != self.eos_id)
            pred = pred[keep]

            timestamp = self.get_timestamps(alphas, peaks, pred)

            preds.append(pred)
            timestamps.append(timestamp)

        return dict(
            preds=preds,
            timestamps=timestamps,
        )

    def encode(self, x: torch.Tensor, seq_lens: torch.Tensor, **kwargs):
        return self.encoder(x, seq_lens)

    def decode(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
        decoder_out = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out = F.log_softmax(decoder_out, dim=-1)
        return decoder_out

    start_end_threshold = 5
    max_token_duration = 12  # 3 times upsampled

    def get_timestamps(self, alphas, peaks, ids, vad_offset=0.0, force_time_shift=-1.5, upsample_rate=3):
        if not len(ids):
            return []

        assert len(alphas.shape) == 1, NotImplementedError('support inference batch_size=1 only')

        if ids[-1] == self.eos_id:
            ids = ids[:-1]

        fire_place = torch.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift  # total offset

        if len(fire_place) != len(ids) + 1:
            # re count the alphas and peaks
            alphas /= alphas.sum() / (len(ids) + 1)
            alphas = alphas.unsqueeze(0)
            peaks = self.neck.cif_wo_hidden(alphas)[0]
            fire_place = torch.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift  # total offset

        timestamps = []
        new_ids = []

        # for bicif model trained with large data, cif2 actually fires when a character starts
        # so treat the frames between two peaks as the duration of the former token
        # begin silence
        if fire_place[0] > self.start_end_threshold:
            # unused, only for debug
            timestamps.append([0.0, fire_place[0]])
            new_ids.append(-1)

        # tokens timestamp
        for i in range(len(fire_place) - 1):
            new_ids.append(ids[i])
            if self.max_token_duration < 0 or fire_place[i + 1] - fire_place[i] <= self.max_token_duration:
                timestamps.append([fire_place[i], fire_place[i + 1]])
            else:
                # cut the duration to token and sil of the 0-weight frames last long
                _split = fire_place[i] + self.max_token_duration
                timestamps.append([fire_place[i], _split])
                # unused, only for debug
                timestamps.append([_split, fire_place[i + 1]])
                new_ids.append(-1)

        num_frames = peaks.shape[0]
        if num_frames - fire_place[-1] > self.start_end_threshold:
            _end = (num_frames + fire_place[-1]) * 0.5
            timestamps[-1][1] = _end
            # unused, only for debug
            timestamps.append([_end, num_frames])
            new_ids.append(-1)
        elif len(timestamps) > 0:
            timestamps[-1][1] = num_frames

        time_rate = 10.0 * 6 / 1000 / upsample_rate
        new_timestamps = []
        for _id, timestamp in zip(new_ids, timestamps):
            if _id == -1:  # <sil>
                continue

            s, e = timestamp
            s, e = float(s), float(e)
            s *= time_rate
            e *= time_rate

            if vad_offset:  # add offset time in model with vad
                s += vad_offset / 1000.0
                e += vad_offset / 1000.0

            s *= 1000
            e *= 1000
            new_timestamps.append([int(s), int(e)])

        return new_timestamps

    def loss(self, encoder_out, encoder_out_mask, speech_lens, text_ids, text_lens, **kwargs):
        """Frontend + Encoder + Decoder + Calc loss"""
        losses = dict()

        # decoder: Attention decoder branch
        loss_att, loss_pre = self.att_loss(encoder_out, speech_lens, encoder_out_mask, text_ids, text_lens)
        loss_pre2 = self.pre2_loss(encoder_out, encoder_out_mask, text_ids, text_lens)

        if self.ctc_weight == 0.0:
            loss = (
                    loss_att
                    + loss_pre * self.predictor_weight
                    + loss_pre2 * self.predictor_weight * 0.5
            )
        else:
            loss_ctc = self._calc_ctc_loss(encoder_out, speech_lens, text_ids, text_lens)

            # Collect CTC branch stats
            losses.update({
                'loss.ctc': loss_ctc
            })

            loss = (
                    self.ctc_weight * loss_ctc
                    + (1 - self.ctc_weight) * loss_att
                    + loss_pre * self.predictor_weight
                    + loss_pre2 * self.predictor_weight * 0.5
            )

        # Collect Attn branch stats
        losses.update({
            'loss': loss,
            'loss.att': loss_att,
            'loss.pre': loss_pre,
            'loss.pre2': loss_pre2,
        })

        return losses

    def att_loss(self, encoder_out, encoder_out_lens, encoder_out_mask, ys_pad, ys_pad_lens):
        pre_acoustic_embeds, pre_token_length, _, _, _ = self.neck(encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)

        # 0. sampler
        if self.sampling_ratio > 0.0:
            sematic_embeds = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds)
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_out = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        return loss_att, loss_pre

    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        tgt_mask = (attentions.make_pad_mask(ys_pad_lens, max_len=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_out = self.decoder(encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens)
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = nonpad_positions.sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
                        value=0,
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)

        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask

    def pre2_loss(self, encoder_out, encoder_out_mask, ys_pad, ys_pad_lens):
        pre_token_length2 = self.neck.get_token_num2(encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id)
        loss_pre2 = self.criterion_pre(ys_pad_lens.type_as(pre_token_length2), pre_token_length2)
        return loss_pre2
