from typing import List

import numpy as np

from utils import os_lib, torch_utils
from . import multimodal_pretrain


class Qwen2VlPredictor(multimodal_pretrain.BaseQwen2Vl):
    def get_model_val_inputs(self, loop_inputs):
        model_inputs = []
        for ret in loop_inputs:
            image = ret['image']
            image_group = ret['image_group']
            text = ret['text']
            video = ret['video']

            content = [
                {
                    "type": "text",
                    "text": text
                }
            ]
            if image is not None:
                content.append({
                    "type": "image",
                    "image": image
                })

            if image_group is not None:
                for image in image_group:
                    content.append({
                        "type": "image",
                        "image": image
                    })

            if video is not None:
                content.append({
                    "type": "video",
                    "video": video
                })

            messages = [{
                "role": "user",
                "content": content,
            },
            ]
            prompt_inputs = self.tokenizer.encode_dialog(messages)
            prompt_inputs['text_ids'] = [prompt_inputs.pop('segment_ids')]
            prompt_inputs['seq_lens'] = [prompt_inputs['seq_lens']]
            if prompt_inputs['image_pixel_values']:
                prompt_inputs['image_pixel_values'] = np.stack(prompt_inputs['image_pixel_values'])
            if prompt_inputs['video_pixel_values']:
                prompt_inputs['video_pixel_values'] = np.stack(prompt_inputs['video_pixel_values'])
            prompt_inputs = torch_utils.Converter.force_to_tensors(prompt_inputs, self.device)
            prompt_inputs['image_pixel_values'] = prompt_inputs['image_pixel_values'].to(self.model.dtype)
            prompt_inputs['video_pixel_values'] = prompt_inputs['video_pixel_values'].to(self.model.dtype)
            model_inputs.append(prompt_inputs)

        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            generated_ids = []
            for model_input in model_inputs:
                model_output = model(**model_input, **model_kwargs)

                preds = model_output['preds']
                preds = preds.cpu().numpy()

                seq_lens = model_input['seq_lens']
                generated_ids += [pred[seq_len:-1] for seq_len, pred in zip(seq_lens, preds)]

            model_results[name] = dict(
                generated_ids=generated_ids,
            )

        return model_results

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, text=None, image=None, image_group=None, video=None, num_pts=4, **kwargs) -> List[dict]:
        if isinstance(image, str):
            image = os_lib.loader.load_img(image)

        if isinstance(video, str):
            video = os_lib.loader.load_video_from_decord(video, num_pts=num_pts)

        rets = []
        for i in range(start_idx, end_idx):
            _text = text[i] if isinstance(text, list) else text
            _image = image[i] if isinstance(image, list) else image
            _image_group = image_group[i] if isinstance(image_group, list) and isinstance(image_group[0], list) else image_group
            _video = video[i] if isinstance(video, list) else video

            if isinstance(_image_group, list):
                for j in range(len(_image_group)):
                    if isinstance(_image_group[j], str):
                        _image_group[j] = os_lib.loader.load_img(_image_group[j])

            rets.append(dict(
                image=_image,
                image_group=_image_group,
                text=_text,
                video=_video,
            ))

        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        generated_ids = model_results[self.model_name]['generated_ids']
        response = self.tokenizer.decode_to_segments(generated_ids)
        process_results.setdefault(self.model_name, []).extend(response)


class Qwen2Vl(multimodal_pretrain.FromQwen2VlPretrained, multimodal_pretrain.DataProcessForQwen2Vl, Qwen2VlPredictor):
    """
    Usage:
        .. code-block:: python

            from bundles.multimodal_pretrain import Qwen2Vl as Process

            model_dir = 'xxx'
            process = Process(
                pretrained_model=model_dir,
                vocab_fn=f'{model_dir}/vocab.json',
                encoder_fn=f'{model_dir}/merges.txt',
                config_version='2b'  # ['2b', '7b', '72b']
            )

            process.init()

            # todo: only support single predict
            text = '描述一下这张图片。'
            image = 'xxx'
            process.single_predict(text=text, image=image)
    """
