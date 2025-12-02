from pathlib import Path
from typing import List

import numpy as np
from torch import nn

from data_parse.nl_data_parse.pre_process import spliter
from processor import Process
from utils import cv_utils, os_lib, configs


class FunAsr(Process):
    """
    Usage:
        from bundles.complex_pipeline import FunAsr as Process

        model_dir = 'xxx'
        processor = Process(
            det_model_dir=f'{model_dir}/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            rec_model_dir=f'{model_dir}/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            punc_model_dir=f'{model_dir}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            spk_model_dir=f'{model_dir}/speech_campplus_sv_zh-cn_16k-common',
        )

        processor.init()

        processor.single_predict(speech_path='xxx')
    """
    model_version = 'FunAsr'
    det_model_dir: str
    rec_model_dir: str
    punc_model_dir: str
    spk_model_dir: str

    det_processor_config = dict()
    rec_processor_config = dict()
    punc_processor_config = dict()
    spk_processor_config = dict()

    use_pretrained = False

    def set_model(self):
        self.set_det_model()
        self.set_rec_model()
        self.set_punc_model()
        self.set_spk_model()

        self.model = nn.Module()  # placeholder

    def set_det_model(self):
        from .speech_detection import DFSMN

        det_processor_config = dict(
            cmvn_path=f'{self.det_model_dir}/am.mvn',
            pretrained_model=f'{self.det_model_dir}/model.pt',
            device=self.device
        )
        det_processor_config = configs.ConfigObjParse.merge_dict(det_processor_config, self.det_processor_config)
        self.det_processor = DFSMN(
            **det_processor_config
        )
        self.det_processor.init()

    def set_rec_model(self):
        from .speech_recognition import BiCifParaformer_Funasr

        rec_processor_config = dict(
            vocab_fn=f'{self.rec_model_dir}/tokens.json',
            seg_dict_path=f'{self.rec_model_dir}/seg_dict',
            cmvn_path=f'{self.rec_model_dir}/am.mvn',
            pretrained_model=f'{self.rec_model_dir}/model.pt',
            device=self.device
        )
        rec_processor_config = configs.ConfigObjParse.merge_dict(rec_processor_config, self.rec_processor_config)
        self.rec_processor = BiCifParaformer_Funasr(
            **rec_processor_config
        )
        self.rec_processor.init()

    def set_punc_model(self):
        from .text_classification import CTTransformer

        punc_processor_config = dict(
            vocab_fn=f'{self.punc_model_dir}/tokens.json',
            pretrained_model=f'{self.punc_model_dir}/model.pt',
            device=self.device
        )
        punc_processor_config = configs.ConfigObjParse.merge_dict(punc_processor_config, self.punc_processor_config)
        self.punc_processor = CTTransformer(
            **punc_processor_config
        )
        self.punc_processor.init()

    def set_spk_model(self):
        from .speech_pretrain import CAMPPlus

        spk_processor_config = dict(
            pretrained_model=f'{self.spk_model_dir}/campplus_cn_common.bin',
            device=self.device
        )
        spk_processor_config = configs.ConfigObjParse.merge_dict(spk_processor_config, self.spk_processor_config)
        self.spk_processor = CAMPPlus(
            **spk_processor_config
        )
        self.spk_processor.init()

    @staticmethod
    def get_chunk_spks(chunk_timestamps, chunk_segments, timestamps_preds):
        sd_time_list = [(spk_st * 1000, spk_ed * 1000, spk) for spk_st, spk_ed, spk in timestamps_preds]
        chunk_spks = []
        for timestamps, segment in zip(chunk_timestamps, chunk_segments):
            sentence_start = timestamps[0][0]
            sentence_end = timestamps[-1][-1]
            sentence_spk = 0
            max_overlap = 0
            for spk_st, spk_ed, spk in sd_time_list:
                overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
                if overlap > max_overlap:
                    max_overlap = overlap
                    sentence_spk = spk
                if overlap > 0 and sentence_spk == spk:
                    max_overlap += overlap
            chunk_spks.append(int(sentence_spk))
        return chunk_spks

    def on_val_step(
            self, loop_objs,
            det_kwargs=dict(), rec_kwargs=dict(), punc_kwargs=dict(), spk_kwargs=dict(),
            rec_batch_size=8, spk_batch_size=16,
            **kwargs
    ) -> dict:
        audio = loop_objs['loop_inputs'][0]['audio']    # todo: only support single predict

        det_outputs = self.det_processor.single_predict(
            speech=audio,
            vis_pbar=False,
            **det_kwargs
        )

        det_timestamps = det_outputs['timestamps']

        batch_speech = []
        for timestamps in det_timestamps:
            bed_idx = int(timestamps[0] * 16)
            end_idx = int(timestamps[1] * 16)
            speech_i = audio[bed_idx:end_idx]
            batch_speech.append(speech_i)

        rec_outputs = self.rec_processor.batch_predict(
            speech=batch_speech,
            total=len(batch_speech),
            batch_size=rec_batch_size,
            vis_pbar=False,
            **rec_kwargs
        )

        segment = []
        timestamps = []
        for i, output in enumerate(rec_outputs):
            segment += output['segment']
            timestamps += [[b + det_timestamps[i][0] for b in a] for a in output['timestamp']]

        punc_outputs = self.punc_processor.single_predict(
            segments=segment,
            vis_pbar=False,
            **punc_kwargs
        )

        chunk_timestamps = []
        chunk_timestamp = []
        for punc_id, timestamp in zip(punc_outputs['punc_ids'], timestamps):
            chunk_timestamp.append(timestamp)
            if punc_id > 1:
                chunk_timestamps.append(chunk_timestamp)
                chunk_timestamp = []

        chunk_segments = []
        chunk_segment = []
        for word in punc_outputs['segment']:
            chunk_segment.append(word)
            if word in self.punc_processor.punc_list:
                chunk_segments.append(chunk_segment)
                chunk_segment = []

        second_timestamps = []
        for timestamp in det_timestamps:
            second_timestamps.append([
                timestamp[0] / 1000.0,
                timestamp[1] / 1000.0,
            ])

        spk_outputs = self.spk_processor.batch_predict(
            speech=batch_speech,
            timestamps=second_timestamps,
            total=len(batch_speech),
            batch_size=spk_batch_size,
            vis_pbar=False,
            **spk_kwargs
        )

        timestamps_preds = spk_outputs['timestamps_preds']

        chunk_spks = self.get_chunk_spks(chunk_timestamps, chunk_segments, timestamps_preds)

        chunk_paragraphs = spliter.ToParagraphs().from_segments_with_zh_en_mix(chunk_segments)
        outputs = []
        for paragraph, segment, timestamp, spk in zip(chunk_paragraphs, chunk_segments, chunk_timestamps, chunk_spks):
            outputs.append(dict(
                paragraph=paragraph,
                segment=segment,
                timestamp=timestamp,
                spk=spk,
            ))

        model_results = {
            self.model_name: outputs
        }
        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        process_results.setdefault(self.model_name, []).append(model_results[self.model_name])

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


class PPOCRv4(Process):
    """
    from bundles.complex_pipeline import PPOCRv4 as Process

    model_dir = 'xxx'
    process = Process(
        det_model_dir=f'{model_dir}/ch_PP-OCRv4_det_server_train',
        rec_model_dir=f'{model_dir}/ch_PP-OCRv4_rec_server_train',
        rec_processor_config=dict(
            vocab_fn=f'{model_dir}/ppocr_keys_v1.txt'
        )
    )
    process.init()

    process.single_predict('xxx.png')
    """
    model_version = 'PPOCRv4'

    det_model_dir: str
    rec_model_dir: str

    det_processor_config = dict()
    rec_processor_config = dict()

    use_pretrained = False

    def set_model(self):
        self.set_det_model()
        self.set_rec_model()

        self.model = nn.Module()  # placeholder

    def set_det_model(self):
        from .object_detection import PPOCRv4Det_Icdar

        det_processor_config = dict(
            pretrained_model=f'{self.det_model_dir}/best_accuracy.pdparams',
            device=self.device
        )
        det_processor_config = configs.ConfigObjParse.merge_dict(det_processor_config, self.det_processor_config)
        self.det_processor = PPOCRv4Det_Icdar(
            **det_processor_config
        )
        self.det_processor.init()

    def set_rec_model(self):
        from .text_recognition import PPOCRv4Rec_MJSynth

        rec_processor_config = dict(
            pretrained_model=f'{self.rec_model_dir}/best_accuracy.pdparams',    # only for teacher model
            device=self.device
        )
        rec_processor_config = configs.ConfigObjParse.merge_dict(rec_processor_config, self.rec_processor_config)
        self.rec_processor = PPOCRv4Rec_MJSynth(
            **rec_processor_config
        )
        self.rec_processor.init()

    def on_val_step(
            self, loop_objs,
            det_kwargs=dict(), rec_kwargs=dict(),
            rec_batch_size=8,
            **kwargs
    ) -> dict:
        image = loop_objs['loop_inputs'][0]['image']    # todo: only support single predict

        det_outputs = self.det_processor.single_predict(image, vis_pbar=False, **det_kwargs)
        segmentations = det_outputs['preds'][0]['segmentations']
        rec_images = []
        for points in segmentations:
            rec_image = cv_utils.ImageCrop.points_to_rectangle(image, points)
            dst_img_height, dst_img_width = rec_image.shape[0:2]
            if dst_img_height / dst_img_width >= 1.5:
                rec_image = np.rot90(image)
            rec_images.append(rec_image)

        rec_outputs = self.rec_processor.batch_predict(rec_images, batch_size=rec_batch_size, vis_pbar=False, **rec_kwargs)
        paragraphs = rec_outputs['preds']

        outputs = []
        for points, paragraph in zip(segmentations, paragraphs):
            outputs.append(dict(
                paragraph=paragraph,
                points=points,
            ))

        model_results = {
            self.model_name: outputs
        }
        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        process_results.setdefault(self.model_name, []).append(model_results[self.model_name])

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        images = objs[0][start_idx: end_idx]
        ids = [Path(image).name if isinstance(image, str) else f'{i}.png' for i, image in zip(range(start_idx, end_idx), images)]
        images = [os_lib.loader.load_img(image, channel_fixed_3=True) if isinstance(image, str) else image for image in images]
        rets = []
        for _id, image in zip(ids, images):
            rets.append(dict(
                _id=_id,
                image=image
            ))
        return rets

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)
