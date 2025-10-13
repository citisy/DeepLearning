from typing import List

import torch

from processor import Process
from utils import os_lib
from .speech_recognition import load_cmvn


class DFSMN(Process):
    model_version = 'DFSMN'
    cmvn_path: str

    def set_model(self):
        from models.speech_detection.DFSMN import Model
        cmvn = load_cmvn(self.cmvn_path)
        cmvn = torch.from_numpy(cmvn)
        self.model = Model(cmvn)

    def load_pretrained(self):
        from models.speech_detection.DFSMN import WeightConverter, WeightLoader

        state_dict = WeightLoader.auto_load(self.pretrained_model, map_location=self.device)
        state_dict = WeightConverter.from_official(state_dict)
        self.model.load_state_dict(state_dict, strict=True)

    def get_model_inputs(self, loop_inputs, train=True):
        return dict(
            audio=torch.from_numpy(loop_inputs[0]['audio']).to(self.device)
        )

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = model_output

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        timestamps = model_results[self.model_name]['timestamps']
        output = process_results.setdefault(self.model_name, dict(
            timestamps=[]
        ))
        output['timestamps'].extend(timestamps)

    sample_rate = 16000

    def gen_predict_inputs(
            self, *objs, start_idx=None, end_idx=None,
            speech=None, speech_path=None,
            **kwargs
    ) -> List[dict]:
        assert end_idx - start_idx == 1, 'Only support batch_size=1'

        if speech is None and speech_path:
            if isinstance(speech_path, str):
                speech_path = [speech_path]
            speech = [os_lib.loader.load_audio(path, self.sample_rate) for path in speech_path]

        if not isinstance(speech, list):
            speech = [None] * start_idx + [speech] * (end_idx - start_idx)

        return [dict(
            audio=speech[i],
        ) for i in range(start_idx, end_idx)]

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)
