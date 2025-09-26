from typing import List

import numpy as np
import torch

from processor import Process
from utils import os_lib


class CAMPPlus(Process):
    model_version = 'CAMPPlus'

    def set_model(self):
        from models.speech_pretrain.CAMPPlus import Model

        self.model = Model()

    def load_pretrained(self):
        if self.pretrain_model:
            from models.speech_pretrain.CAMPPlus import Model, WeightConverter

            tensor = torch.load(self.pretrain_model, map_location=torch.device('cpu'))
            tensor = WeightConverter.from_official(tensor)

            self.model.load_state_dict(tensor, strict=True)

    def chunk_speech(self, speech, timestamps, seg_dur=1.5, seg_shift=0.75):
        chunk_timestamps = []
        chunk_speech = []
        chunk_idx = []
        for i, (audio, timestamp) in enumerate(zip(speech, timestamps)):
            seg_st = timestamp[0]
            chunk_len = int(seg_dur * self.sample_rate)
            chunk_shift = int(seg_shift * self.sample_rate)
            last_chunk_ed = 0
            for chunk_st in range(0, audio.shape[0], chunk_shift):
                chunk_ed = min(chunk_st + chunk_len, audio.shape[0])
                if chunk_ed <= last_chunk_ed:
                    break
                last_chunk_ed = chunk_ed
                chunk_st = max(0, chunk_ed - chunk_len)
                chunk_data = audio[chunk_st:chunk_ed]
                if chunk_data.shape[0] < chunk_len:
                    chunk_data = np.pad(chunk_data, (0, chunk_len - chunk_data.shape[0]), "constant")
                chunk_timestamps.append([chunk_st / self.sample_rate + seg_st, chunk_ed / self.sample_rate + seg_st])
                chunk_speech.append(chunk_data)
                chunk_idx.append(i)

        return chunk_speech, chunk_timestamps, chunk_idx

    def get_model_inputs(self, loop_inputs, train=True):
        speech = [ret['audio'] for ret in loop_inputs]
        timestamps = [ret['timestamp'] for ret in loop_inputs]
        chunk_speech, chunk_timestamps, chunk_idx = self.chunk_speech(speech, timestamps)
        chunk_speech = np.stack(chunk_speech, axis=0)
        chunk_speech = torch.from_numpy(chunk_speech).to(self.device)
        model_inputs = dict(
            speech=chunk_speech,
            timestamps=chunk_timestamps,
        )
        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), batch_size=None, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)
        speech = model_inputs['speech']

        model_results = {}
        for name, model in self.models.items():
            spk_embeddings = []
            # the data will be overlarge the batch_size, so that re-batch the data again
            for i in range(0, len(speech), batch_size):
                _model_inputs = dict(
                    speech=speech[i:i + batch_size],
                    **model_kwargs
                )

                model_outputs = model(**_model_inputs)
                spk_embeddings.append(model_outputs['hidden'])

            spk_embeddings = torch.cat(spk_embeddings, dim=0)
            spk_embeddings = spk_embeddings.cpu()

            model_results[name] = dict(
                spk_embeddings=spk_embeddings,
                timestamps=model_inputs['timestamps']
            )

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        spk_embeddings = model_results[self.model_name]['spk_embeddings']
        timestamps = model_results[self.model_name]['timestamps']
        output = process_results.setdefault(self.model_name, dict(
            spk_embeddings=[],
            timestamps=[]
        ))
        output['spk_embeddings'].extend(spk_embeddings)
        output['timestamps'].extend(timestamps)

    def on_val_end(self, loop_objs, process_results=dict(), **kwargs):
        output = process_results[self.model_name]
        spk_embeddings = output['spk_embeddings']
        spk_embeddings = torch.stack(spk_embeddings)
        timestamps = output['timestamps']
        outputs = self.model.post_process(spk_embeddings, timestamps)
        outputs.update(
            spk_embeddings=spk_embeddings,
        )
        return outputs

    sample_rate = 16000

    def gen_predict_inputs(
            self, *objs, start_idx=None, end_idx=None,
            speech=None, speech_path=None,
            timestamps=None,
            **kwargs
    ) -> List[dict]:
        if speech is None and speech_path:
            if isinstance(speech_path, str):
                speech_path = [speech_path]
            speech = [os_lib.loader.load_audio(path, self.sample_rate) for path in speech_path]

        if not isinstance(speech, list):
            speech = [None] * start_idx + [speech] * (end_idx - start_idx)

        if isinstance(timestamps, list) and not isinstance(timestamps[0], list):
            timestamps = [None] * start_idx + [timestamps] * (end_idx - start_idx)

        inputs = []
        for i in range(start_idx, end_idx):
            per_input = dict(
                audio=speech[i],
                timestamp=timestamps[i]
            )
            inputs.append(per_input)

        return inputs

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)

    def on_predict_end(self, *args, **kwargs):
        return self.on_val_end(*args, **kwargs)
