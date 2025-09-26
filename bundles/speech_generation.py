from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from data_parse.nl_data_parse.pre_process import chunker, snack
from processor import Process
from utils import configs, math_utils, os_lib, torch_utils


class CosyVoice(Process):
    """
    Usage:
        processor = CosyVoice(
            model_dir='xxx/CosyVoice-300M-Instruct'
        )
        processor.init()

        query_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
        prompt_text = '希望你以后能够做的比我还好呦。'
        prompt_speech_path = 'xxx/zero_shot_prompt.wav'

        # zero_shot mode
        processor.single_predict(
            query_text=query_text,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
            save_path=f'{processor.cache_dir}/zero_shot.wav'
        )

        # cross_lingual mode
        query_text2 = '<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.'
        processor.single_predict(
            query_text=query_text2,
            prompt_speech_path=prompt_speech_path,
            save_path=f'{processor.cache_dir}/cross_lingual.wav'
        )

        # vc mode
        source_speech_path = 'xxx/cross_lingual_prompt.wav'
        processor.single_predict(
            source_speech_path=source_speech_path,
            prompt_speech_path=prompt_speech_path,
            save_path=f'{processor.cache_dir}/vc.wav'
        )

        # sft mode
        processor.single_predict(
            query_text=query_text,
            spk_id='中文女',
            save_path=f'{processor.cache_dir}/sft.wav'
        )

        # instruct mode
        query_text3 = '在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。'
        prompt_text2 = 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.'
        processor.single_predict(
            query_text=query_text3,
            spk_id='中文男',
            prompt_text=prompt_text2,
            is_instruct=True,
            save_path=f'{processor.cache_dir}/instruct.wav'
        )
    """
    model_version = 'CosyVoice'

    model_dir: str

    def set_model(self):
        from models.speech_generation.CosyVoice import Model

        self.model = Model(
            front_config=dict(
                campplus_model=f'{self.model_dir}/campplus.onnx',
                speech_idsizer_model=f'{self.model_dir}/speech_tokenizer_v1.onnx',
            )
        )

    use_pretrained = True

    def load_pretrained(self):
        if self.use_pretrained:
            from models.speech_generation.CosyVoice import WeightConverter

            llm_state_dict = torch.load(f'{self.model_dir}/llm.pt')
            llm_state_dict = WeightConverter.from_official(llm_state_dict)
            self.model.llm.load_state_dict(llm_state_dict, strict=True, assign=True)

            flow_state_dict = torch.load(f'{self.model_dir}/flow.pt')
            flow_state_dict = WeightConverter.from_official(flow_state_dict)
            self.model.flow.load_state_dict(flow_state_dict, strict=True, assign=True)

            hift_state_dict = torch.load(f'{self.model_dir}/hift.pt')
            hift_state_dict = WeightConverter.from_official(hift_state_dict)
            self.model.hift.load_state_dict(hift_state_dict, strict=True, assign=True)

            spk2info = torch.load(f'{self.model_dir}/spk2info.pt')
            self.model.front.update_spk2info(spk2info)

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import WhisperTokenizer
        self.tokenizer = WhisperTokenizer()

    max_prompt_speech_len = 15  # seconds

    def get_model_inputs(self, loop_inputs, train=True) -> dict:
        if train:
            raise NotImplementedError

        model_inputs = []
        for i, ret in enumerate(loop_inputs):
            is_instruct = ret.get('is_instruct', False)

            prompt_text = ret.get('prompt_text')
            prompt_speech = ret.get('prompt_speech')
            if prompt_text:
                assert is_instruct or prompt_speech.shape[1] <= self.max_prompt_speech_len * 16000, \
                    f'prompt_speech must be less than {self.max_prompt_speech_len}s, if want to break the limit, try to run without `prompt_text` or use instruct mode'
                if is_instruct:
                    prompt_text += '<|endofprompt|>'

                prompt_inputs = self.tokenizer.encode_paragraphs([prompt_text])

                prompt_inputs = dict(
                    prompt_text_ids=prompt_inputs['segments_ids'][0],
                    prompt_text_ids_len=prompt_inputs['seq_lens'][0],
                    prompt_speech=prompt_speech[0]
                )
            else:
                # vc mode
                prompt_inputs = dict(
                    prompt_speech=prompt_speech[0][:self.max_prompt_speech_len * 16000]
                )

            source_speech = ret.get('source_speech')
            if source_speech is None:
                # zero shot mode
                spk_id = ret.get('spk_id')
                query_text = ret.get('query_text')
                query_texts = chunker.RetentionToChunkedParagraphs(max_len=80).from_paragraph(query_text)
                query_texts = snack.add_end_token(query_texts)
                query_inputs = self.tokenizer.encode_paragraphs(query_texts, pad_type=0)
                for text_ids in query_inputs['segments_ids']:
                    per_model_input = dict(
                        text_ids=text_ids,
                        text_ids_len=len(text_ids),
                        spk_id=spk_id,
                        is_instruct=is_instruct,
                        batch_idx=i,
                        **prompt_inputs
                    )
                    per_model_input = torch_utils.Converter.force_to_tensors(per_model_input, device=self.device)
                    model_inputs.append(per_model_input)

            else:
                # vc mode
                per_model_input = dict(
                    source_speech=source_speech[0],
                    is_instruct=is_instruct,
                    batch_idx=i,
                    **prompt_inputs
                )

                per_model_input = torch_utils.Converter.force_to_tensors(per_model_input, device=self.device)
                model_inputs.append(per_model_input)

        list_keys = ['prompt_text_ids', 'prompt_text_ids_len', 'prompt_speech', 'text_ids', 'text_ids_len', 'batch_idx']
        _model_inputs = dict()
        for per_model_input in model_inputs:
            for k, v in per_model_input.items():
                if k in list_keys:
                    _model_inputs.setdefault(k, []).append(v)
                else:
                    vv = _model_inputs.setdefault(k, v)
                    assert vv == v

        for k, v in _model_inputs.items():
            if k in ['prompt_text_ids', 'prompt_speech', 'text_ids']:
                _model_inputs[k] = pad_sequence(v, batch_first=True, padding_value=0)
            elif k in ['prompt_text_ids_len', 'text_ids_len']:
                _model_inputs[k] = torch.tensor(v, device=self.device)
        return _model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), batch_size=None, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        batch_idx = model_inputs.pop('batch_idx')

        model_results = {}
        for name, model in self.models.items():
            query_speeches = []
            # the data will be overlarge the batch_size, so that re-batch the data again
            for i in range(0, len(batch_idx), batch_size):
                _model_inputs = dict()
                for k, v in model_inputs.items():
                    if k in ['prompt_text_ids', 'prompt_text_ids_len', 'prompt_speech', 'text_ids', 'text_ids_len']:
                        _model_inputs[k] = v[i:i + batch_size]
                    else:
                        _model_inputs[k] = v

                _model_inputs.update(model_kwargs)

                model_outputs = model(**_model_inputs)
                query_speeches += model_outputs['hidden']

            query_speeches = math_utils.unique_gather(batch_idx, query_speeches)
            query_speeches = [torch.cat(q, dim=1).cpu() for q in query_speeches]

            model_results[name] = dict(
                query_speeches=query_speeches,
            )

        return model_results

    def gen_predict_inputs(
            self, *objs, start_idx=None, end_idx=None,
            prompt_speech=None, prompt_speech_path=None,
            source_speech=None, source_speech_path=None,
            **kwargs
    ) -> List[dict]:
        if prompt_speech is None and prompt_speech_path:
            prompt_speech, sr = os_lib.loader.load_audio_from_torchaudio(prompt_speech_path, backend='soundfile')
            prompt_speech = prompt_speech.mean(dim=0, keepdim=True)
            if sr != 16000:
                assert sr > 16000, f'wav sample rate {sr} must be greater than 16000'
                prompt_speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(prompt_speech)

        if source_speech is None and source_speech_path:
            source_speech, _ = os_lib.loader.load_audio_from_torchaudio(source_speech_path)

        inputs = []
        keys = ['query_text', 'prompt_text', 'save_path']
        for i in range(start_idx, end_idx):
            per_input = dict(
                prompt_speech=prompt_speech,
                source_speech=source_speech
            )
            for k, v in kwargs.items():
                if k in keys and not isinstance(v, str):
                    per_input[k] = v[i]
                else:
                    per_input[k] = v

            inputs.append(per_input)

        return inputs

    def on_predict_step_end(self, loop_objs, **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        for ret, query_speech in zip(loop_inputs, model_results[self.model_name]['query_speeches']):
            save_path = ret['save_path']
            os_lib.mk_parent_dir(save_path)
            torchaudio.save(save_path, query_speech, self.model.front.sample_rate)


class CosyVoice2(CosyVoice):
    """
    Usage:
        processor = CosyVoice2(
            model_dir='xxx/CosyVoice2-0___5B'
        )
        processor.init()

        query_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
        prompt_text = '希望你以后能够做的比我还好呦。'
        prompt_speech_path = 'xxx/zero_shot_prompt.wav'

        # zero_shot mode
        processor.single_predict(
            query_text=query_text,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
            save_path=f'{processor.cache_dir}/zero_shot.wav'
        )

        # fine_grained_control mode
        query_text2 = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
        processor.single_predict(
            query_text=query_text2,
            prompt_speech_path=prompt_speech_path,
            save_path=f'{processor.cache_dir}/fine_grained_control.wav'
        )

        # instruct mode
        prompt_text2 = '用四川话说这句话'
        processor.single_predict(
            query_text=query_text,
            prompt_text=prompt_text2,
            prompt_speech_path=prompt_speech_path,
            is_instruct=True,
            save_path=f'{processor.cache_dir}/instruct.wav'
        )
    """
    model_version = 'CosyVoice2'
    config_version = '0.5b'

    def set_model(self):
        from models.speech_generation.CosyVoice2 import Model, Config

        cfgs = Config.get(self.config_version)
        sp_cfg = dict(
            front_config=dict(
                campplus_model=f'{self.model_dir}/campplus.onnx',
                speech_idsizer_model=f'{self.model_dir}/speech_tokenizer_v2.onnx',
            ),
        )
        cfgs = configs.ConfigObjParse.merge_dict(cfgs, sp_cfg)

        self.model = Model(**cfgs)

    def load_pretrained(self):
        from models.speech_generation.CosyVoice2 import WeightConverter

        llm_state_dict = torch.load(f'{self.model_dir}/llm.pt')
        llm_state_dict = WeightConverter.from_official(llm_state_dict)
        self.model.llm.load_state_dict(llm_state_dict, strict=True, assign=True)

        flow_state_dict = torch.load(f'{self.model_dir}/flow.pt')
        flow_state_dict = WeightConverter.from_official(flow_state_dict)
        self.model.flow.load_state_dict(flow_state_dict, strict=True, assign=True)

        hift_state_dict = torch.load(f'{self.model_dir}/hift.pt')
        hift_state_dict = WeightConverter.from_official(hift_state_dict)
        self.model.hift.load_state_dict(hift_state_dict, strict=True, assign=True)

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import Qwen2Tokenizer

        sp_token_dict = dict(
            eos="<|endoftext|>",
            pad="<|endoftext|>",
            im_start='<|im_start|>',
            im_end='<|im_end|>',
            endofprompt='<|endofprompt|>',
            breath='[breath]',
            strong_start='<strong>',
            strong_end='</strong>',
            noise='[noise]',
            laughter='[laughter]',
            cough='[cough]',
            clucking='[clucking]',
            accent='[accent]',
            quick_breath='[quick_breath]',
            laughter_start='<laughter>',
            laughter_end='</laughter>',
            hissing='[hissing]',
            sigh='[sigh]',
            vocalized_noise='[vocalized-noise]',
            lipsmack='[lipsmack]',
            mn='[mn]'
        )

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            f'{self.model_dir}/CosyVoice-BlankEN/vocab.json',
            f'{self.model_dir}/CosyVoice-BlankEN/merges.txt',
        )
        self.tokenizer.update_sp_token(sp_token_dict)
