import copy
import math
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

from utils import torch_utils


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            'encoder.in_linear1.linear': 'encoder.to_in.0',
            'encoder.in_linear2.linear': 'encoder.to_in.1',
            'encoder.out_linear1.linear': 'encoder.to_out.0',
            'encoder.out_linear2.linear': 'encoder.to_out.1',
            'encoder.fsmn.{0}.linear.linear': 'encoder.fsmn.{0}.linear',
            'encoder.fsmn.{0}.affine.linear': 'encoder.fsmn.{0}.affine',
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class VadStateMachine(Enum):
    StartPoint = 1
    InSpeechSegment = 2
    EndPoint = 3


class FrameState(Enum):
    Invalid = -1
    Speech = 1
    Silence = 0


class AudioChangeState(Enum):
    Speech2Speech = 0
    Speech2Silence = 1
    Silence2Silence = 2
    Silence2Speech = 3
    NoBegin = 4
    Invalid = 5


class VadDetectMode(Enum):
    SingleDetect = 0
    MutipleDetect = 1


class Stats:
    def __init__(self):
        self.data_buf_start_frame = 0
        self.frm_cnt = 0
        self.latest_confirmed_speech_frame = 0
        self.latest_confirmed_silence_frame = -1
        self.continuous_silence_frame_count = 0
        self.vad_state_machine = VadStateMachine.StartPoint
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.number_end_time_detected = 0
        self.noise_average_decibel = -100.0
        self.next_seg = True

        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.scores = None
        self.decibel = []
        self.data_buf = None
        self.data_buf_all = None
        self.last_drop_frames = 0


class WindowDetector:
    def __init__(
            self,
            window_size_ms: int,
            sil_to_speech_time: int,
            speech_to_sil_time: int,
            frame_size_ms: int,
    ):
        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame

        self.cur_win_pos = 0
        self.pre_frame_state = FrameState.Silence
        self.cur_frame_state = FrameState.Silence
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)

        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def detect_one_frame(self, frame_state: FrameState) -> AudioChangeState:
        if frame_state == FrameState.Speech:
            cur_frame_state = 1
        elif frame_state == FrameState.Silence:
            cur_frame_state = 0
        else:
            return AudioChangeState.Invalid

        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
                self.pre_frame_state == FrameState.Silence
                and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.Speech
            return AudioChangeState.Silence2Speech

        elif (
                self.pre_frame_state == FrameState.Speech
                and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.Silence
            return AudioChangeState.Speech2Silence

        elif self.pre_frame_state == FrameState.Silence:
            return AudioChangeState.Silence2Silence

        elif self.pre_frame_state == FrameState.Speech:
            return AudioChangeState.Speech2Speech

        else:
            return AudioChangeState.Invalid

    def reset(self) -> None:
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.Silence
        self.cur_frame_state = FrameState.Silence
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0


class E2EVadSpeechBufWithDoa:
    def __init__(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0


class Model(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    sample_rate: int = 16000
    detect_mode: int = VadDetectMode.MutipleDetect
    snr_mode: int = 0
    max_end_silence_time: int = 800
    max_start_silence_time: int = 3000
    do_start_point_detection: bool = True
    do_end_point_detection: bool = True
    window_size_ms: int = 200
    sil_to_speech_time_thres: int = 150
    speech_to_sil_time_thres: int = 150
    speech_2_noise_ratio: float = 1.0
    do_extend: bool = True
    lookback_time_start_point: int = 200
    lookahead_time_end_point: int = 100
    max_single_segment_time: int = 60000
    dcd_block_size: int = 4
    snr_thres: int = -100.0
    noise_frame_num_used_for_snr: int = 100
    decibel_thres: int = -100.0
    silence_pdf_num: int = 1
    speech_noise_thresh_low: float = -0.1
    speech_noise_thresh_high: float = 0.3
    frame_in_ms: int = 10
    frame_length_ms: int = 25
    speech_noise_thres: float = 0.6
    sil_pdf_ids: List[int] = [0]

    def __init__(self, cmvn, frontend_config={}, encoder_config={}, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.frontend = WavFrontendOnline(cmvn, **frontend_config)
        self.encoder = FSMN(**encoder_config)

    def init_caches(self, max_end_silence_time=None, **kwargs):
        if max_end_silence_time is not None:
            # update the max_end_silence_time
            self.max_end_silence_time = max_end_silence_time
        self.max_end_sil_frame_cnt_thresh = self.max_end_silence_time - self.speech_to_sil_time_thres

        windows_detector = WindowDetector(
            self.window_size_ms,
            self.sil_to_speech_time_thres,
            self.speech_to_sil_time_thres,
            self.frame_in_ms,
        )

        stats = Stats()
        return dict(
            frontend={},
            prev_samples=torch.empty(0),
            encoder={},
            windows_detector=windows_detector,
            stats=stats
        )

    def forward(self, x, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.post_process(x, **kwargs)

    def post_process(self, x, is_final=True, chunk_size=60000, caches=None, **kwargs):
        if caches is None:
            caches = self.init_caches()

        chunk_stride_samples = int(chunk_size * self.sample_rate / 1000)
        is_streaming_input = False if chunk_size >= 15000 else True
        n_chunk = int(len(x) // chunk_stride_samples + int(is_final))

        segments = []
        for i in range(n_chunk):
            _is_final = is_final and i == n_chunk - 1
            audio_sample_i = x[i * chunk_stride_samples: (i + 1) * chunk_stride_samples]

            speech = audio_sample_i[None]

            chunk_segment = self.detect_one_chunk(
                speech,
                caches=caches,
                is_final=_is_final,
                is_streaming_input=is_streaming_input
            )
            segments += chunk_segment

        return dict(
            segments=segments,
            caches=caches
        )

    def detect_one_chunk(
            self,
            speech: torch.Tensor,
            caches=None,
            is_final: bool = True,
            is_streaming_input=True,
            **kwargs,
    ):
        feats, feats_lengths, waveform = self.frontend(speech, caches=caches['frontend'])

        stats = caches["stats"]
        windows_detector = caches["windows_detector"]

        if stats.data_buf_all is None:
            stats.data_buf_all = waveform[0]
            stats.data_buf = stats.data_buf_all
        else:
            stats.data_buf_all = torch.cat((stats.data_buf_all, waveform[0]))

        decibel = self.compute_decibel(waveform)
        stats.decibel.extend(decibel)

        scores = self.encoder(feats, caches["encoder"])  # return B * T * D
        stats.nn_eval_block_size = scores.shape[1]
        stats.frm_cnt += scores.shape[1]  # count total frames
        if stats.scores is None:
            stats.scores = scores  # the first calculation
        else:
            stats.scores = torch.cat((stats.scores, scores), dim=1)

        if not is_final:
            self.detect_common_frames(stats, windows_detector)
        else:
            self.detect_last_frames(stats, windows_detector)

        segments = self.gen_segments(stats, is_streaming_input, is_final)
        return segments

    def gen_segments(self, stats, is_streaming_input, is_final):
        segments = []
        if len(stats.output_data_buf) > 0:
            for i in range(stats.output_data_buf_offset, len(stats.output_data_buf)):
                data_buf = stats.output_data_buf[i]
                if is_streaming_input:  # in this case, return [beg, -1], [], [-1, end], [beg, end]
                    assert data_buf.contain_seg_start_point
                    assert stats.next_seg or data_buf.contain_seg_end_point

                    start_ms = data_buf.start_ms if stats.next_seg else -1

                    if data_buf.contain_seg_end_point:
                        end_ms = data_buf.end_ms
                        stats.next_seg = True
                        stats.output_data_buf_offset += 1
                    else:
                        end_ms = -1
                        stats.next_seg = False

                else:  # in this case, return [beg, end]
                    assert is_final or (data_buf.contain_seg_start_point and data_buf.contain_seg_end_point)

                    start_ms = data_buf.start_ms
                    end_ms = data_buf.end_ms
                    stats.output_data_buf_offset += 1  # need update this parameter

                segment = [start_ms, end_ms]
                segments.append(segment)

        return segments

    def compute_decibel(self, waveform,):
        frame_sample_length = int(self.frame_length_ms * self.sample_rate / 1000)
        frame_shift_length = int(self.frame_in_ms * self.sample_rate / 1000)

        offsets = torch.arange(0, waveform.shape[1] - frame_sample_length + 1, frame_shift_length)
        frames = waveform[0, offsets[:, None] + torch.arange(frame_sample_length)]

        decibel = 10 * torch.log10(torch.sum(torch.square(frames), dim=1) + 0.000001)
        decibel = decibel.numpy().tolist()
        return decibel

    def detect_last_frames(self, stats, windows_detector):
        if stats.vad_state_machine == VadStateMachine.EndPoint:
            return

        for i in range(stats.nn_eval_block_size)[::-1]:
            frame_state = self.get_frame_state(stats.frm_cnt - 1 - i - stats.last_drop_frames, stats, windows_detector)
            self.detect_one_frame(frame_state, stats.frm_cnt - 1 - i, i == 0, stats, windows_detector)

    def detect_common_frames(self, stats, windows_detector):
        if stats.vad_state_machine == VadStateMachine.EndPoint:
            return

        for i in range(stats.nn_eval_block_size)[::-1]:
            frame_state = self.get_frame_state(stats.frm_cnt - 1 - i - stats.last_drop_frames, stats, windows_detector)
            self.detect_one_frame(frame_state, stats.frm_cnt - 1 - i, False, stats, windows_detector)

    def get_frame_state(self, t: int, stats, windows_detector):
        cur_decibel = stats.decibel[t]
        cur_snr = cur_decibel - stats.noise_average_decibel
        if cur_decibel < self.decibel_thres:
            frame_state = FrameState.Silence
            self.detect_one_frame(frame_state, t, False, stats, windows_detector)
            return frame_state

        else:
            sum_score = 0.0
            noise_prob = 0.0
            if len(self.sil_pdf_ids) > 0:
                if len(self.sil_pdf_ids) > 1:
                    sum_score = sum(stats.scores[0][t][sil_pdf_id].item() for sil_pdf_id in self.sil_pdf_ids)
                else:
                    sum_score = stats.scores[0][t][self.sil_pdf_ids[0]].item()
                noise_prob = math.log(sum_score) * self.speech_2_noise_ratio
                total_score = 1.0
                sum_score = total_score - sum_score
            speech_prob = math.log(sum_score)

            if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
                if cur_snr >= self.snr_thres and cur_decibel >= self.decibel_thres:
                    frame_state = FrameState.Speech
                else:
                    frame_state = FrameState.Silence
            else:
                frame_state = FrameState.Silence
                if stats.noise_average_decibel < -100:
                    stats.noise_average_decibel = cur_decibel
                else:
                    stats.noise_average_decibel = (cur_decibel + stats.noise_average_decibel * (self.noise_frame_num_used_for_snr - 1)) / self.noise_frame_num_used_for_snr

            return frame_state

    def detect_one_frame(
            self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool, stats, windows_detector
    ) -> None:
        state_change = windows_detector.detect_one_frame(cur_frm_state)

        frm_shift_in_ms = self.frame_in_ms
        if state_change == AudioChangeState.Silence2Speech:
            stats.continuous_silence_frame_count = 0
            if stats.vad_state_machine == VadStateMachine.StartPoint:
                start_frame = max(
                    stats.data_buf_start_frame,
                    cur_frm_idx - self.latency_frm_num_at_start_point(windows_detector),
                )
                self.on_voice_start(start_frame, stats)
                stats.vad_state_machine = VadStateMachine.InSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.on_voice_detected(t, stats)
            elif stats.vad_state_machine == VadStateMachine.InSpeechSegment:
                for t in range(stats.latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.on_voice_detected(t, stats)

                if cur_frm_idx - stats.confirmed_start_frame + 1 > self.max_single_segment_time / frm_shift_in_ms:
                    self.on_voice_end(cur_frm_idx, False, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, stats)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, stats)
            else:
                pass

        elif state_change == AudioChangeState.Speech2Silence:
            stats.continuous_silence_frame_count = 0
            if stats.vad_state_machine == VadStateMachine.StartPoint:
                pass
            elif stats.vad_state_machine == VadStateMachine.InSpeechSegment:
                if (
                        cur_frm_idx - stats.confirmed_start_frame + 1
                        > self.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, stats)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, stats)

        elif state_change == AudioChangeState.Speech2Speech:
            stats.continuous_silence_frame_count = 0
            if stats.vad_state_machine == VadStateMachine.InSpeechSegment:
                if (
                        cur_frm_idx - stats.confirmed_start_frame + 1
                        > self.max_single_segment_time / frm_shift_in_ms
                ):
                    # stats.max_time_out = True
                    self.on_voice_end(cur_frm_idx, False, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, stats)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, stats)

        elif state_change == AudioChangeState.Silence2Silence:
            stats.continuous_silence_frame_count += 1
            if stats.vad_state_machine == VadStateMachine.StartPoint:
                # silence timeout, return zero length decision
                if (
                        self.detect_mode == VadDetectMode.SingleDetect
                        and stats.continuous_silence_frame_count * frm_shift_in_ms > self.max_start_silence_time
                ) or (
                        is_final_frame and stats.number_end_time_detected == 0
                ):
                    for t in range(stats.latest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.on_silence_detected(t, stats)
                    self.on_voice_start(0, stats, True)
                    self.on_voice_end(0, True, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint
                else:
                    if cur_frm_idx >= self.latency_frm_num_at_start_point(windows_detector):
                        self.on_silence_detected(
                            cur_frm_idx - self.latency_frm_num_at_start_point(windows_detector),
                            stats
                        )

            elif stats.vad_state_machine == VadStateMachine.InSpeechSegment:
                if stats.continuous_silence_frame_count * frm_shift_in_ms >= self.max_end_sil_frame_cnt_thresh:
                    lookback_frame = int(self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms)
                    if self.do_extend:
                        lookback_frame -= int(self.lookahead_time_end_point / frm_shift_in_ms)
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.on_voice_end(cur_frm_idx - lookback_frame, False, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint

                elif cur_frm_idx - stats.confirmed_start_frame + 1 > self.max_single_segment_time / frm_shift_in_ms:
                    self.on_voice_end(cur_frm_idx, False, False, stats)
                    stats.vad_state_machine = VadStateMachine.EndPoint

                elif self.do_extend and not is_final_frame:
                    if stats.continuous_silence_frame_count <= int(self.lookahead_time_end_point / frm_shift_in_ms):
                        self.on_voice_detected(cur_frm_idx, stats)

                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, stats)

        if (
                stats.vad_state_machine == VadStateMachine.EndPoint
                and self.detect_mode == VadDetectMode.MutipleDetect
        ):
            self.reset_detection(stats, windows_detector)

    def latency_frm_num_at_start_point(self, windows_detector) -> int:
        vad_latency = windows_detector.win_size_frame
        if self.do_extend:
            vad_latency += int(self.lookback_time_start_point / self.frame_in_ms)
        return vad_latency

    def on_voice_start(self, start_frame: int, stats, fake_result: bool = False) -> None:
        if stats.confirmed_start_frame == -1:
            stats.confirmed_start_frame = start_frame

        if (
                not fake_result
                and stats.vad_state_machine == VadStateMachine.StartPoint
        ):
            self.pop_data_to_output_buf(
                stats.confirmed_start_frame, 1, True, False, False, stats
            )

    def on_voice_detected(self, valid_frame: int, stats) -> None:
        stats.latest_confirmed_speech_frame = valid_frame
        self.pop_data_to_output_buf(valid_frame, 1, False, False, False, stats)

    def on_voice_end(
            self, end_frame: int, fake_result: bool, is_last_frame: bool, stats
    ) -> None:
        for t in range(stats.latest_confirmed_speech_frame + 1, end_frame):
            self.on_voice_detected(t, stats)

        if stats.confirmed_end_frame == -1:
            stats.confirmed_end_frame = end_frame

        if not fake_result:
            self.pop_data_to_output_buf(stats.confirmed_end_frame, 1, False, True, is_last_frame, stats)
        stats.number_end_time_detected += 1

    def pop_data_to_output_buf(
            self,
            start_frm: int,
            frm_cnt: int,
            first_frm_is_start_point: bool,
            last_frm_is_end_point: bool,
            end_point_is_sent_end: bool,
            stats
    ) -> None:
        self.pop_data_buf_till_frame(start_frm, stats)
        expected_sample_number = int(frm_cnt * self.sample_rate * self.frame_in_ms / 1000)
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(self.frame_length_ms * self.sample_rate / 1000 - self.sample_rate * self.frame_in_ms / 1000),
            )
            expected_sample_number += int(extra_sample)

        if len(stats.output_data_buf) == 0 or first_frm_is_start_point:
            data_buf = E2EVadSpeechBufWithDoa()
            data_buf.start_ms = start_frm * self.frame_in_ms
            data_buf.end_ms = data_buf.start_ms
            data_buf.doa = 0
            stats.output_data_buf.append(data_buf)
        cur_seg = stats.output_data_buf[-1]

        cur_seg.doa = 0
        stats.data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def pop_data_buf_till_frame(self, frame_idx: int, stats) -> None:  # need check again
        while stats.data_buf_start_frame < frame_idx:
            if len(stats.data_buf) >= int(self.frame_in_ms * self.sample_rate / 1000):
                stats.data_buf_start_frame += 1
                stats.data_buf = stats.data_buf_all[(stats.data_buf_start_frame - stats.last_drop_frames) * int(self.frame_in_ms * self.sample_rate / 1000):]

    def maybe_on_voice_end_if_last_frame(
            self, is_final_frame: bool, cur_frm_idx: int, stats
    ) -> None:
        if is_final_frame:
            self.on_voice_end(cur_frm_idx, False, True, stats)
            stats.vad_state_machine = VadStateMachine.EndPoint

    def reset_detection(self, stats, windows_detector):
        stats.continuous_silence_frame_count = 0
        stats.latest_confirmed_speech_frame = 0
        stats.latest_confirmed_silence_frame = -1
        stats.confirmed_start_frame = -1
        stats.confirmed_end_frame = -1
        stats.vad_state_machine = VadStateMachine.StartPoint
        windows_detector.reset()

        if stats.output_data_buf:
            drop_frames = int(stats.output_data_buf[-1].end_ms / self.frame_in_ms)
            real_drop_frames = drop_frames - stats.last_drop_frames
            stats.last_drop_frames = drop_frames
            stats.data_buf_all = stats.data_buf_all[real_drop_frames * int(self.frame_in_ms * self.sample_rate / 1000):]
            stats.decibel = stats.decibel[real_drop_frames:]
            stats.scores = stats.scores[:, real_drop_frames:, :]

    def on_silence_detected(self, valid_frame: int, stats):
        stats.latest_confirmed_silence_frame = valid_frame
        if stats.vad_state_machine == VadStateMachine.StartPoint:
            self.pop_data_buf_till_frame(valid_frame, stats)


class WavFrontendOnline(nn.Module):
    """Conventional frontend structure for streaming ASR/VAD."""

    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    filter_length_min: int = -1
    filter_length_max: int = -1
    lfr_m: int = 5
    lfr_n: int = 1
    dither: float = 0.0
    snip_edges: bool = True
    upsacle_samples: bool = True

    def __init__(self, cmvn, **kwargs):
        super().__init__()
        self.frame_sample_length = int(self.frame_length * self.fs / 1000)
        self.frame_shift_sample_length = int(self.frame_shift * self.fs / 1000)
        self.register_buffer('cmvn', cmvn, persistent=False)

    def forward(self, x, is_final=False, caches={}):
        reserve_waveforms = caches['reserve_waveforms'] if 'reserve_waveforms' in caches else torch.empty(0)
        input_cache = caches['input_cache'] if 'input_cache' in caches else torch.empty(0)
        lfr_splice_cache = caches['lfr_splice_cache'] if 'lfr_splice_cache' in caches else []

        batch_size = x.shape[0]

        waveforms, feats, feats_lengths, input_cache, fbanks, fbanks_lens = self.forward_fbank(x, input_cache)  # input shape: B T D

        if feats.shape[0]:
            waveforms = torch.cat((reserve_waveforms, waveforms), dim=1)

            if not lfr_splice_cache:  # 初始化splice_cache
                for i in range(batch_size):
                    lfr_splice_cache.append(
                        feats[i][0, :].unsqueeze(dim=0).repeat((self.lfr_m - 1) // 2, 1)
                    )
            # need the number of the input frames + self.lfr_splice_cache[0].shape[0] is greater than self.lfr_m
            if feats_lengths[0] + lfr_splice_cache[0].shape[0] >= self.lfr_m:
                lfr_splice_cache_tensor = torch.stack(lfr_splice_cache)  # B T D
                feats = torch.cat((lfr_splice_cache_tensor, feats), dim=1)
                feats_lengths += lfr_splice_cache_tensor[0].shape[0]
                frame_from_waveforms = int(
                    (waveforms.shape[1] - self.frame_sample_length)
                    / self.frame_shift_sample_length
                    + 1
                )
                minus_frame = (self.lfr_m - 1) // 2 if reserve_waveforms.numel() == 0 else 0
                feats, feats_lengths, lfr_splice_frame_idxs = self.forward_lfr_cmvn(feats, feats_lengths, is_final, lfr_splice_cache)
                if self.lfr_m == 1:
                    reserve_waveforms = torch.empty(0)
                else:
                    reserve_frame_idx = lfr_splice_frame_idxs[0] - minus_frame
                    reserve_waveforms = waveforms[:, reserve_frame_idx * self.frame_shift_sample_length: frame_from_waveforms * self.frame_shift_sample_length, ]
                    sample_length = (frame_from_waveforms - 1) * self.frame_shift_sample_length + self.frame_sample_length
                    waveforms = waveforms[:, :sample_length]
            else:
                # update self.reserve_waveforms and self.lfr_splice_cache
                reserve_waveforms = waveforms[:, : -(self.frame_sample_length - self.frame_shift_sample_length)]
                for i in range(batch_size):
                    lfr_splice_cache[i] = torch.cat((lfr_splice_cache[i], feats[i]), dim=0)
                feats = torch.empty(0)
        else:
            if is_final:
                waveforms = (
                    waveforms
                    if reserve_waveforms.numel() == 0
                    else reserve_waveforms
                )
                feats = torch.stack(lfr_splice_cache)
                feats_lengths = torch.zeros(batch_size, dtype=torch.int) + feats.shape[1]
                feats, feats_lengths, _ = self.forward_lfr_cmvn(feats, feats_lengths, is_final, lfr_splice_cache)

        caches.update(
            reserve_waveforms=reserve_waveforms,
            input_cache=input_cache,
            lfr_splice_cache=lfr_splice_cache
        )

        return feats, feats_lengths, waveforms

    def forward_fbank(self, x, input_cache):
        batch_size = x.size(0)

        x = torch.cat((input_cache, x), dim=1)
        frame_num = self.compute_frame_num(x.shape[-1])
        # update self.in_cache
        input_cache = x[:, -(x.shape[-1] - frame_num * self.frame_shift_sample_length):]
        waveforms = torch.empty(0)
        feats_pad = torch.empty(0)
        feats_lens = torch.empty(0)
        if frame_num:
            waveforms = []
            feats = []
            feats_lens = []
            for i in range(batch_size):
                waveform = x[i]
                # we need accurate wave samples that used for fbank extracting
                waveforms.append(
                    waveform[:(frame_num - 1) * self.frame_shift_sample_length + self.frame_sample_length]
                )
                waveform = waveform * (1 << 15)
                waveform = waveform.unsqueeze(0)
                feat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.n_mels,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=self.dither,
                    energy_floor=0.0,
                    window_type=self.window,
                    sample_frequency=self.fs,
                )

                feat_length = feat.size(0)
                feats.append(feat)
                feats_lens.append(feat_length)

            waveforms = torch.stack(waveforms)
            feats_lens = torch.as_tensor(feats_lens)
            feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        fbanks = feats_pad
        fbanks_lens = copy.deepcopy(feats_lens)
        return waveforms, feats_pad, feats_lens, input_cache, fbanks, fbanks_lens

    def forward_lfr_cmvn(
            self,
            x: torch.Tensor,
            input_lengths: torch.Tensor,
            is_final: bool = False,
            lfr_splice_cache=[],
    ):
        batch_size = x.size(0)
        feats = []
        feats_lens = []
        lfr_splice_frame_idxs = []
        for i in range(batch_size):
            feat = x[i, : input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                feat, lfr_splice_cache[i], lfr_splice_frame_idx = self.apply_lfr(feat, is_final)
            feat = self.apply_cmvn(feat)
            feat_length = feat.size(0)
            feats.append(feat)
            feats_lens.append(feat_length)
            lfr_splice_frame_idxs.append(lfr_splice_frame_idx)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
        lfr_splice_frame_idxs = torch.as_tensor(lfr_splice_frame_idxs)
        return feats_pad, feats_lens, lfr_splice_frame_idxs

    def apply_cmvn(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CMVN with mvn data"""
        frame, dim = inputs.shape

        cmvn = self.cmvn.to(inputs)
        means = np.tile(cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(cmvn[1:2, :dim], (frame, 1))
        inputs += means
        inputs *= vars

        return inputs.type(torch.float32)

    def apply_lfr(self, x: torch.Tensor, is_final: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply lfr with data
        """
        lfr_m = self.lfr_m
        lfr_n = self.lfr_n

        T = x.shape[0]  # include the right context
        T_lfr = int(
            np.ceil((T - (lfr_m - 1) // 2) / lfr_n)
        )  # minus the right context: (lfr_m - 1) // 2
        splice_idx = T_lfr
        feat_dim = x.shape[-1]
        ori_inputs = x
        strides = (lfr_n * feat_dim, 1)
        sizes = (T_lfr, lfr_m * feat_dim)
        last_idx = (T - lfr_m) // lfr_n + 1
        num_padding = lfr_m - (T - last_idx * lfr_n)
        if is_final:
            if num_padding > 0:
                num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
                x = torch.vstack([x] + [x[-1:]] * int(num_padding))
        else:
            if num_padding > 0:
                sizes = (last_idx, lfr_m * feat_dim)
                splice_idx = last_idx
        splice_idx = min(T - 1, splice_idx * lfr_n)
        LFR_outputs = x[:splice_idx].as_strided(sizes, strides)
        lfr_splice_cache = ori_inputs[splice_idx:, :]
        return LFR_outputs.clone().type(torch.float32), lfr_splice_cache, splice_idx

    def compute_frame_num(self, sample_length: int) -> int:
        frame_num = int((sample_length - self.frame_sample_length) / self.frame_shift_sample_length + 1)
        return frame_num if frame_num >= 1 and sample_length >= self.frame_sample_length else 0


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3: (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3: (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.from_numpy(cmvn)
    return cmvn


class FSMN(nn.Module):
    def __init__(
            self,
            input_dim: int = 400,
            input_affine_dim: int = 140,
            fsmn_layers: int = 4,
            linear_dim: int = 250,
            proj_dim: int = 128,
            lorder: int = 20,
            rorder: int = 0,
            lstride: int = 1,
            rstride: int = 0,
            output_affine_dim: int = 140,
            output_dim: int = 248,
            use_softmax: bool = True,
    ):
        super().__init__()
        self.to_in = nn.Sequential(
            # note, emmmmm, it looks like so unreasonable that there's no act func between linear layers
            nn.Linear(input_dim, input_affine_dim),
            nn.Linear(input_affine_dim, linear_dim),
            nn.ReLU()
        )
        self.fsmn = nn.ModuleList([
            BasicBlock(linear_dim, proj_dim, lorder, rorder, lstride, rstride, i)
            for i in range(fsmn_layers)
        ])
        self.to_out = nn.Sequential(
            nn.Linear(linear_dim, output_affine_dim),
            nn.Linear(output_affine_dim, output_dim),
            nn.Softmax(dim=-1) if use_softmax else nn.Identity()
        )

    def forward(self, x, caches=None):
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, D)
            caches: when cache is not None, the forward is in streaming. The type of cache is a dict, egs,
            {'cache_layer_1': torch.Tensor(B, T1, D)}, T1 is equal to self.lorder. It is {} for the 1st frame
        """

        x = self.to_in(x)

        for module in self.fsmn:
            # self.cache will update automatically in self.fsmn
            x = module(x, caches)

        x = self.to_out(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            linear_dim: int,
            proj_dim: int,
            lorder: int,
            rorder: int,
            lstride: int,
            rstride: int,
            stack_layer: int,
    ):
        super().__init__()
        self.lorder = lorder
        self.lstride = lstride
        self.stack_layer = stack_layer

        self.linear = nn.Linear(linear_dim, proj_dim, bias=False)

        self.fsmn_block = FSMNBlock(proj_dim, proj_dim, lorder, rorder, lstride, rstride)

        self.affine = nn.Linear(proj_dim, linear_dim)
        self.relu = nn.ReLU()

    def forward(self, x, caches=None):
        x = self.linear(x)  # B T D

        if caches is not None:
            if self.stack_layer not in caches:
                caches[self.stack_layer] = torch.zeros(
                    x.shape[0], x.shape[-1], (self.lorder - 1) * self.lstride, 1
                ).to(x)
            x, caches[self.stack_layer] = self.fsmn_block(x, caches[self.stack_layer])
        else:
            x, _ = self.fsmn_block(x, None)

        x = self.affine(x)
        x = self.relu(x)
        return x


class FSMNBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            lorder=None,
            rorder=None,
            lstride=1,
            rstride=1,
    ):
        super().__init__()
        self.lorder = lorder
        self.lstride = lstride

        self.conv_left = nn.Conv2d(
            input_dim, input_dim, [lorder, 1], dilation=[lstride, 1], groups=input_dim, bias=False
        )

        if rorder > 0:
            self.conv_right = nn.Conv2d(
                input_dim, input_dim, [rorder, 1], dilation=[rstride, 1], groups=input_dim, bias=False
            )
        else:
            self.conv_right = None

    def forward(self, x: torch.Tensor, cache: torch.Tensor = None):
        x = torch.unsqueeze(x, 1).permute(0, 3, 2, 1)  # B D T C

        if cache is not None:
            y_left = torch.cat((cache, x), dim=2)
            cache = y_left[:, :, -(self.lorder - 1) * self.lstride:, :]
        else:
            y_left = F.pad(x, [0, 0, (self.lorder - 1) * self.lstride, 0])

        y_left = self.conv_left(y_left)
        y = x + y_left

        if self.conv_right is not None:
            # maybe need to check
            y_right = F.pad(x, [0, 0, 0, self.rorder * self.rstride])
            y_right = y_right[:, :, self.rstride:, :]
            y_right = self.conv_right(y_right)
            y += y_right

        y = y.permute(0, 3, 2, 1).squeeze(1)
        return y, cache
