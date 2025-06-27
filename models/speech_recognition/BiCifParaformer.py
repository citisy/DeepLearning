import torch
import torch.nn.functional as F

from . import Paraformer
from .. import attentions


class Model(Paraformer.Model):
    """refer to:
    - [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
    - [Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model](https://arxiv.org/abs/2301.12343)
    """

    def post_process(self, x, seq_lens=None, **kwargs):
        x, seq_lens = self.frontend(x, seq_lens)

        encoder_out, seq_lens = self.encode(x, seq_lens)

        encoder_out_mask = attentions.make_pad_mask(seq_lens, max_len=encoder_out.size(1))[:, None, :].to(encoder_out.device)

        # predict the length of token
        pre_acoustic_embeds, token_num, alphas, us_alphas, us_peaks = self.neck(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)

        token_num = token_num.round().long()
        if torch.max(token_num) < 1:
            return []

        decoder_out = self.decode(encoder_out, seq_lens, pre_acoustic_embeds, token_num)

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            alphas = us_alphas[i][: seq_lens[i] * self.neck.upsample_times]
            peaks = us_peaks[i][: seq_lens[i] * self.neck.upsample_times]

            am_scores = decoder_out[i, : token_num[i], :]
            preds = am_scores.argmax(dim=-1)
            keep = (preds != self.blank_id) & (preds != self.sos_id) & (preds != self.eos_id)
            preds = preds[keep]

            timestamps = self.get_timestamps(alphas, peaks, preds)

            results.append({
                "preds": preds,
                'timestamps': timestamps,
            })

        return results

    def encode(self, x: torch.Tensor, seq_lens: torch.Tensor, **kwargs):
        """
        Args:
            x: (Batch, Length, ...)
            seq_lens: (Batch, )
        """
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
