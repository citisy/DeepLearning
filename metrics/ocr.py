import numpy as np
import pandas as pd
from tqdm import tqdm
from . import object_detection, text_generation
from data_parse.nl_data_parse.pre_process import spliter
from utils import os_lib


class EasyMetric:
    def __init__(self, iou_thres=0.5, verbose=True, stdout_method=print, **ap_kwargs):
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
        self.ap = object_detection.AP(**ap_kwargs)

    def get_det_rets(self, gt_iter_data, det_iter_data, image_dir=None):
        rets = {}
        for ret in tqdm(gt_iter_data, desc='load gt data'):
            dic = rets.setdefault(ret['_id'], {})
            dic['gt_boxes'] = ret['bboxes']
            dic['image_dir'] = ret['image_dir'] if 'image_dir' in ret else image_dir
            if 'confs' in ret:
                dic['gt_confs'] = ret['confs']

        for ret in tqdm(det_iter_data, desc='load det data'):
            dic = rets[ret['_id']]
            dic['det_boxes'] = ret['bboxes']
            dic['confs'] = np.ones(len(ret['bboxes']))

        gt_boxes = [v['gt_boxes'] for v in rets.values()]
        det_boxes = [v['det_boxes'] for v in rets.values()]
        confs = [v['confs'] for v in rets.values()]
        _ids = list(rets.keys())

        return rets, _ids, gt_boxes, det_boxes, confs

    def get_rec_rets(self, gt_iter_data, det_iter_data, image_dir=None):
        rets = {}
        for ret in tqdm(gt_iter_data):
            dic = rets.setdefault(ret['_id'], {})
            dic['true'] = ret['transcription']
            dic['image_dir'] = ret['image_dir'] if 'image_dir' in ret else image_dir

        for ret in tqdm(det_iter_data):
            dic = rets[ret['_id']]
            dic['pred'] = ret['transcription']

        det_text = [v['pred'] for v in rets.values()]
        gt_text = [v['true'] for v in rets.values()]
        _ids = list(rets.keys())

        return rets, _ids, gt_text, det_text

    def det_quick_metrics(self, gt_iter_data, det_iter_data, save_path=None):
        """

        Args:
            gt_iter_data:
            det_iter_data:
            save_path:

        Returns:

        Usage:
            .. code-block:: python

                # use ppocr type data result to metric
                from cv_data_parse.PaddleOcr import Loader, DataRegister
                data_dir = 'your data dir'

                loader = Loader(data_dir)
                gt_iter_data = loader.load_det(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task='gt set task')
                det_iter_data = loader.load_det(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task='det set task')

                det_quick_metrics(gt_iter_data, det_iter_data)
        """
        rets, _ids, gt_boxes, det_boxes, confs = self.get_det_rets(gt_iter_data, det_iter_data)
        ret = object_detection.ap.mAP_thres_range(gt_boxes, det_boxes, confs)
        df = pd.DataFrame(ret)
        df = df.round(4)
        self.stdout_method(df)

        if save_path:
            os_lib.Saver(verbose=self.verbose, stdout_method=self.stdout_method).auto_save(df, save_path)

        return df

    def rec_quick_metrics(self, gt_iter_data, det_iter_data, save_path=None):
        """

        Args:
            gt_iter_data:
            det_iter_data:
            save_path:

        Returns:

        Usage:
            .. code-block:: python

                # use ppocr type data result to metric
                from cv_data_parse.PaddleOcr import Loader, DataRegister
                data_dir = 'your data dir'
                gt_set_task = 'gt_rec'
                det_rec_task = 'det_rec'

                loader = Loader(data_dir)
                gt_iter_data = loader.load_rec(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task=gt_set_task)
                det_iter_data = loader.load_rec(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task=det_rec_task)

                rec_quick_metrics(gt_iter_data, det_res_path)
        """
        from utils import nlp_utils

        rets, _ids, gt_text, det_text = self.get_rec_rets(gt_iter_data, det_iter_data)
        ret = {}

        _ret = {
            'char': text_generation.TopMetric(confusion_method=text_generation.CharConfusionMatrix),
            'line': text_generation.TopMetric(confusion_method=text_generation.LineConfusionMatrix)
        }

        ret.update({k: v.f_measure(det_text, gt_text) for k, v in _ret.items()})

        _ret = {
            'ROUGE-1': text_generation.TopMetric(confusion_method=text_generation.WordConfusionMatrix, n_gram=1),
            'ROUGE-2': text_generation.TopMetric(confusion_method=text_generation.WordConfusionMatrix, n_gram=2),
            'ROUGE-3': text_generation.TopMetric(confusion_method=text_generation.WordConfusionMatrix, n_gram=3),
            'ROUGE-L': text_generation.TopMetric(confusion_method=text_generation.WordLCSConfusionMatrix),
            'ROUGE-W': text_generation.TopMetric(confusion_method=text_generation.WordLCSConfusionMatrix, lcs_method=nlp_utils.Sequencer.weighted_longest_common_subsequence),
        }

        det_cut_text = spliter.segments_from_paragraphs_by_jieba(det_text)
        gt_cut_text = spliter.segments_from_paragraphs_by_jieba(gt_text)

        ret.update({k: v.f_measure(det_cut_text, gt_cut_text) for k, v in _ret.items()})

        df = pd.DataFrame(ret).T
        df = df.round(6)
        self.stdout_method(df)

        if save_path:
            os_lib.Saver(verbose=self.verbose, stdout_method=self.stdout_method).auto_save(df, save_path)

        return df

    def det_checkout_false_sample(self, gt_iter_data, det_iter_data, data_dir='checkout_data', image_dir=None, save_res_dir=None):
        from utils import os_lib, visualize
        from data_parse.cv_data_parse.base import DataVisualizer

        image_dir = image_dir if image_dir is not None else f'{data_dir}/images'
        save_res_dir = save_res_dir if save_res_dir is not None else f'{data_dir}/visuals/false_samples'

        rets, _ids, gt_boxes, det_boxes, confs = self.get_det_rets(gt_iter_data, det_iter_data, image_dir=image_dir)
        self.ap.return_more_info = True
        ret = self.ap.mAP_thres(gt_boxes, det_boxes, confs)
        r = ret['']
        tp = r['tp']
        det_obj_idx = r['det_obj_idx']
        target_obj_idx = det_obj_idx[~tp]

        idx = np.unique(target_obj_idx)
        for i in idx:
            target_idx = det_obj_idx == i
            _tp = tp[target_idx]

            gt_box = gt_boxes[i]
            det_box = det_boxes[i]
            _id = _ids[i]
            image = os_lib.loader.load_img(f'{rets[_id]["image_dir"]}/{_id}')

            false_obj_idx = np.where(~_tp)[0]

            gt_colors = [visualize.get_color_array(0) for _ in gt_box]
            det_colors = [visualize.get_color_array(0) for _ in det_box]
            for _ in false_obj_idx:
                det_colors[_] = visualize.get_color_array(-1)

            tmp_gt = [dict(_id=_id, image=image, bboxes=gt_box, colors=gt_colors)]
            tmp_det = [dict(image=image, bboxes=det_box, colors=det_colors)]
            visualizer = DataVisualizer(save_res_dir, verbose=self.verbose, stdout_method=self.stdout_method)
            visualizer(tmp_gt, tmp_det)

        return ret

    def rec_checkout_false_sample(self, gt_iter_data, det_iter_data, data_dir='checkout_data', image_dir=None, save_res_dir=None):
        from utils import nlp_utils, os_lib, visualize

        image_dir = image_dir if image_dir is not None else f'{data_dir}/images'
        save_res_dir = save_res_dir if save_res_dir is not None else f'{data_dir}/visuals/false_samples'
        rets, _ids, gt_text, det_text = self.get_rec_rets(gt_iter_data, det_iter_data, image_dir=image_dir)

        cm = text_generation.LineConfusionMatrix()

        tp = np.array(cm.tp(det_text, gt_text)['tp'])
        cp = np.array(cm.cp(gt_text)['cp'])
        idx = np.where(tp != cp)[0]

        ret = []
        for i in idx:
            _id = _ids[i]
            image = os_lib.loader.load_img(f'{image_dir}/{_id}')

            gt_image = np.zeros_like(image) + 255
            text_boxes = [[(4, 4), (image.shape[1], 4), (image.shape[1], image.shape[0]), (4, image.shape[0])]]
            texts = [gt_text[i]]
            gt_image = visualize.ImageVisualize.text(gt_image, text_boxes, texts)

            det_image = np.zeros_like(image) + 255
            text_boxes = [[(4, 4), (image.shape[1], 4), (image.shape[1], image.shape[0]), (4, image.shape[0])]]
            texts = [det_text[i]]
            det_image = visualize.ImageVisualize.text(det_image, text_boxes, texts)

            image = np.concatenate([image, gt_image, det_image], axis=0)
            os_lib.Saver(verbose=self.verbose, stdout_method=self.stdout_method).auto_save(image, f'{save_res_dir}/{_id}')

            ret.append(dict(
                _id=_id,
                gt=gt_text[i],
                det=det_text[i]
            ))
            self.stdout_method(gt_text[i])
            self.stdout_method(det_text[i])
            self.stdout_method('------------')

        os_lib.saver.save_json(ret, f'{save_res_dir}/result.json')
        return ret
