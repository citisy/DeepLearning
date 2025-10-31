import torch
import torch.nn.functional as F
from torch import nn

from . import Paraformer
from ..bundles import WeightLoader
from .. import attentions


class WeightConverter(Paraformer.WeightConverter):
    pass


class Model(Paraformer.Model):
    """refer to:
    paper:
        - [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
        - [Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model](https://arxiv.org/abs/2301.12343)
    """

    def make_neck(self, **head_config):
        self.neck = CifPredictorV3(**head_config)

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
        loss_pre2 = self.pre2_loss(encoder_out, encoder_out_mask, text_lens)

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

    def pre2_loss(self, encoder_out, encoder_out_mask, ys_pad_lens):
        pre_token_length2 = self.neck.get_token_num2(encoder_out, encoder_out_mask)
        loss_pre2 = self.criterion_pre(ys_pad_lens.type_as(pre_token_length2), pre_token_length2)
        return loss_pre2


class CifPredictorV3(nn.Module):
    """Continuous integrate-and-fire(Cif)
    code: `funasr.models.bicif_paraformer.cif_predictor.CifPredictorV3`"""
    input_dim = 512
    l_order = 1
    r_order = 1
    threshold = 1.0
    drop_prob = 0.1
    smooth_factor = 1.0
    noise_threshold = 0
    tail_threshold = 0.45
    smooth_factor2 = 0.25
    noise_threshold2 = 0.01
    upsample_times = 3
    use_cif1_cnn = False

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        input_dim = self.input_dim
        l_order = self.l_order
        r_order = self.r_order

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(input_dim, input_dim, l_order + r_order + 1)
        self.cif_output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.upsample_cnn = nn.ConvTranspose1d(input_dim, input_dim, self.upsample_times, self.upsample_times)
        self.blstm = nn.LSTM(input_dim, input_dim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.cif_output2 = nn.Linear(input_dim * 2, 1)

    def get_token_num2(self, hidden, mask=None):
        context = hidden.transpose(1, 2)  # (b, s, d) -> (b, d, s)
        queries = self.pad(context)
        output = F.relu(self.cif_conv1d(queries))

        # alphas2 is an extra head for timestamp prediction
        if not self.use_cif1_cnn:
            _output = context
        else:
            _output = output

        output2 = self.upsample_cnn(_output).transpose(1, 2)
        output2, (_, _) = self.blstm(output2)

        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask2 = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(alphas2.shape[0], -1)
            )
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        token_num2 = alphas2.sum(-1)
        return token_num2

    def forward(
            self,
            hidden,
            target_label=None,
            mask=None,
            ignore_id=-1,
            mask_chunk_predictor=None,
            target_label_length=None,
    ):
        h = hidden
        mk = mask

        context = hidden.transpose(1, 2)  # (b, s, d) -> (b, d, s)
        queries = self.pad(context)
        output = F.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = F.sigmoid(output)
        alphas = F.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mk is not None:
            mk = mk.transpose(-1, -2).float()
            alphas = alphas * mk
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mk = mk.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            h, alphas, token_num = self.tail_process_fn(h, alphas, mask=mk)

        acoustic_embeds = self.cif(h, alphas)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        us_alphas, us_peaks = self.get_timestamp(hidden, mask, token_num)
        return acoustic_embeds, token_num, alphas, us_alphas, us_peaks

    def tail_process_fn(self, hidden, alphas, mask=None):
        b, s, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def cif(self, hidden, alphas):
        """count embeddings"""
        b, s, d = hidden.shape

        # loop vars
        integrate = torch.zeros([b], device=hidden.device)
        frame = torch.zeros([b, d], device=hidden.device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(s):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([b], device=hidden.device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= self.threshold
            integrate = torch.where(
                fire_place,
                integrate - torch.ones([b], device=hidden.device),
                integrate
            )
            cur = torch.where(fire_place, distribution_completion, alpha)

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(
                fire_place[:, None].repeat(1, d),
                (alpha - cur)[:, None] * hidden[:, t, :],
                frame
            )

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        embeds = []
        token_num = torch.round(alphas.sum(-1)).int()
        max_token_num = token_num.max()
        for i in range(b):
            l = torch.index_select(frames[i], 0, torch.nonzero(fires[i] >= self.threshold).squeeze())
            # use zero embed to the last frame if not completed.
            pad_l = torch.zeros([max_token_num - l.size(0), d], device=hidden.device)
            embeds.append(torch.cat([l, pad_l], 0))
        return torch.stack(embeds, 0)

    def get_timestamp(self, hidden, mask=None, token_num=None):
        context = hidden.transpose(1, 2)

        # an extra head for timestamp prediction
        if self.use_cif1_cnn:
            queries = self.pad(context)
            output = F.relu(self.cif_conv1d(queries))
        else:
            output = context

        output = self.upsample_cnn(output).transpose(1, 2)
        output, (_, _) = self.blstm(output)

        us_alphas = F.sigmoid(self.cif_output2(output))
        us_alphas = F.relu(us_alphas * self.smooth_factor2 - self.noise_threshold2)
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(us_alphas.shape[0], -1)
                .unsqueeze(-1)
            )  # (b, 1, s) -> (b, self.upsample_times * s, 1)
            us_alphas = us_alphas * mask
        us_alphas = us_alphas.squeeze(-1)
        if token_num is not None:
            _token_num = us_alphas.sum(-1)
            us_alphas *= (token_num / _token_num)[:, None].repeat(1, us_alphas.size(1))

        # upsampled alphas and cif_peak
        us_cif_peak = self.cif_wo_hidden(us_alphas)
        return us_alphas, us_cif_peak

    def cif_wo_hidden(self, alphas):
        threshold = self.threshold - 1e-4
        batch_size, len_time = alphas.size()

        # loop varss
        integrate = torch.zeros([batch_size], device=alphas.device)
        # intermediate vars along time
        list_fires = []

        for t in range(len_time):
            alpha = alphas[:, t]

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            integrate = torch.where(
                fire_place,
                integrate - torch.ones([batch_size], device=alphas.device) * threshold,
                integrate,
            )

        fires = torch.stack(list_fires, 1)
        return fires
