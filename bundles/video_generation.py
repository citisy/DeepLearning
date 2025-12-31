from datetime import datetime
from typing import List

import imageio
import torch
import torchvision

from processor import Process
from utils import os_lib, torch_utils


class Wan2_1(Process):
    """
    from bundles.video_generation import Wan2_1 as Process

    model_dir = 'xxx'
    process = Process(
        t5_text_encoder_pretrained=f'{model_dir}/models_t5_umt5-xxl-enc-bf16.pth',
        vae_pretrained=f'{model_dir}/Wan2.1_VAE.pth',
        backbone_pretrained=f'{model_dir}/diffusion_pytorch_model.safetensors',

        vocab_fn=f'{model_dir}/google/umt5-xxl/tokenizer.json',
        encoder_fn=f'{model_dir}/google/umt5-xxl/spiece.model',
    )
    process.init()
    process.single_predict(
        'xxx',
        neg_texts='xxx',

        model_kwargs=dict(
            vis_pbar=True,
        ),
        is_visualize=True
    )

    """
    low_memory_run = True
    use_half = True

    config_version = 't2v-1.3b'
    model_version = 'wan2.1'
    dataset_version = ''

    def set_model(self):
        from models.video_generation.wan2_1 import Model, Config

        with torch.device('meta'):
            self.model = Model(**Config.get(self.config_version))

    clip_text_encoder_pretrained: List[str] | str
    t5_text_encoder_pretrained: List[str] | str
    backbone_pretrained: List[str] | str
    vae_pretrained: List[str] | str

    def load_pretrained(self):
        from models.video_generation.wan2_1 import WeightConverter, Config
        from models.bundles import WeightLoader

        state_dicts = {
            "t5": WeightLoader.auto_load(self.t5_text_encoder_pretrained),
            "backbone": WeightLoader.auto_load(self.backbone_pretrained),
            "vae": WeightLoader.auto_load(self.vae_pretrained),
        }

        if self.model.model_type != Config.ModelType.t2v:
            state_dicts.update(
                clip=WeightLoader.auto_load(self.clip_text_encoder_pretrained)
            )

        state_dict = WeightConverter.from_official(state_dicts)
        self.model.load_state_dict(state_dict, strict=False, assign=True)

        self.log(f'Loaded pretrained model!')

    def set_model_status(self):
        if self.use_pretrained:
            self.load_pretrained()
        if self.low_memory_run:
            self.model._device = self.device  # explicitly define the device for the model
            self.model.set_low_memory_run()

        else:
            if not isinstance(self.device, list):
                self.model.to(self.device)

        if self.use_half:
            self.model.set_half()
        else:
            self.model.to(torch.float)

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process import bundled

        self.tokenizer = bundled.T5Tokenizer.from_pretrained(
            vocab_fn=self.vocab_fn,
            encoder_fn=self.encoder_fn
        )

    def gen_predict_inputs(self, *objs, neg_texts='', start_idx=None, end_idx=None, **kwargs):
        pos_texts = objs[0]
        if not isinstance(neg_texts, list):
            neg_texts = [neg_texts] * len(pos_texts)

        rets = []
        for i in range(start_idx, end_idx):
            rets.append(dict(
                text=pos_texts[i],
                neg_text=neg_texts[i]
            ))

        return rets

    def get_model_inputs(self, loop_inputs, train=True):
        texts = []
        neg_texts = []
        for ret in loop_inputs:
            if 'text' in ret:
                texts.append(ret['text'])
            if 'neg_text' in ret:
                neg_texts.append(ret['neg_text'])

        inputs = self.tokenizer.encode_paragraphs(texts, pad_type=2)
        neg_inputs = self.tokenizer.encode_paragraphs(neg_texts, pad_type=2)

        model_inputs = dict(
            text_ids=inputs['segments_ids'],
            text_mask=inputs['valid_segment_tags'],
            neg_text_ids=neg_inputs['segments_ids'],
            neg_text_mask=neg_inputs['valid_segment_tags'],
        )

        model_inputs = torch_utils.Converter.force_to_tensors(model_inputs, self.device)

        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            video = model(**model_inputs)
            model_results[name] = dict(
                video=video,
            )

        return model_results

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for model_name, results in model_results.items():
            r = process_results.setdefault(model_name, dict())
            for data_name, items in results.items():
                r.setdefault(data_name, []).extend(items)

    def on_val_step_end(self, loop_objs, is_visualize=False, save_to_one_dir=True, **kwargs):
        model_results = loop_objs['model_results']

        if is_visualize:
            sub_dir, sub_name = self._make_save_obj(save_to_one_dir)
            for model_name, results in model_results.items():
                for data_name, video in results.items():
                    cache_dir = f'{self.cache_dir}/{sub_dir}/{model_name}'
                    video_save_stem = f'{sub_name}{data_name}'
                    self.visualize_one(video, cache_dir, video_save_stem, **kwargs)

    def _make_save_obj(self, save_to_one_dir):
        date = str(datetime.now().isoformat(timespec='seconds', sep=' '))
        if save_to_one_dir:
            sub_dir = ''
            sub_name = date + '.'
        else:
            sub_dir = date
            sub_name = ''

        return sub_dir, sub_name

    def visualize_one(self, video, cache_dir, video_save_stem='', **kwargs):
        cache_dir = f'{cache_dir}'
        os_lib.mk_dir(cache_dir)

        value_range = (-1, 1)
        nrow = 8
        normalize = True
        fps = 16
        video = video.clamp(min(value_range), max(value_range))
        video = torch.stack([torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in video.unbind(2)], dim=1).permute(1, 2, 3, 0)
        video = (video * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(f'{cache_dir}/{video_save_stem}.mp4', fps=fps, codec='libx264', quality=8)
        for frame in video.numpy():
            writer.append_data(frame)
        writer.close()
