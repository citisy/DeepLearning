import json
import pandas as pd
from tqdm import tqdm
from . import object_detection, text_generation
from utils import os_lib


def det_quick_metrics(gt_iter_data, det_iter_data, save_path=None):
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
            det_iter_data = loader.load_det(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task='det set task', label_dir='visuals')

            det_quick_metrics(gt_iter_data, det_iter_data)
    """
    r = {}

    for ret in tqdm(gt_iter_data):
        r.setdefault(ret['_id'], {})['gt_boxes'] = ret['bboxes']

    for ret in tqdm(det_iter_data):
        r.setdefault(ret['_id'], {})['det_boxes'] = ret['bboxes']
        r.setdefault(ret['_id'], {})['confs'] = [1] * len(ret['bboxes'])

    gt_boxes = [v['gt_boxes'] for v in r.values()]
    det_boxes = [v['det_boxes'] for v in r.values()]
    confs = [v['confs'] for v in r.values()]

    ret = object_detection.ap.mAP_thres_range(gt_boxes, det_boxes, confs)
    df = pd.DataFrame(ret)
    df = df.round(4)
    print(df)

    if save_path:
        os_lib.saver.auto_save(df, save_path)

    return df


def rec_quick_metrics(gt_iter_data, det_res_path, save_path=None):
    """

    Args:
        gt_iter_data:
        det_res_path:
        save_path:

    Returns:

    Usage:
        .. code-block:: python

            # use ppocr type data result to metric
            from cv_data_parse.PaddleOcr import Loader, DataRegister
            data_dir = 'your data dir'

            loader = Loader(data_dir)
            gt_iter_data = loader.load_rec(set_type=DataRegister.TEST, image_type=DataRegister.PATH, set_task='rec')
            det_res_path = 'your res path'

            rec_quick_metrics(gt_iter_data, det_res_path)
    """
    from utils import nlp_utils

    r = {}
    for ret in tqdm(gt_iter_data):
        image = ret['image']
        transcription = ret['transcription']
        r.setdefault(image, {})['true'] = transcription

    with open(det_res_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            image, ret = line.split('\t')
            ret = json.loads(ret)
            transcription = ret['Student']['label']
            r.setdefault(image, {})['pred'] = transcription

    det_text = [ret['pred'] for ret in r.values()]
    gt_text = [ret['true'] for ret in r.values()]

    ret = {}

    _ret = {
        'char': text_generation.TopMetric(confusion_method=text_generation.CharConfusionMatrix),
        'line': text_generation.TopMetric(confusion_method=text_generation.LineConfusionMatrix)
    }

    for v in _ret.values():
        v.return_more_info = True

    ret.update({k: v.f_measure(det_text, gt_text) for k, v in _ret.items()})

    _ret = {
        'ROUGE-2': text_generation.TopMetric(confusion_method=text_generation.WordConfusionMatrix, n_gram=2, is_cut=True),
        'ROUGE-3': text_generation.TopMetric(confusion_method=text_generation.WordConfusionMatrix, n_gram=3, is_cut=True),
        'ROUGE-L': text_generation.TopMetric(confusion_method=text_generation.WordLCSConfusionMatrix, is_cut=True),
        'ROUGE-W': text_generation.TopMetric(confusion_method=text_generation.WordLCSConfusionMatrix, lcs_method=nlp_utils.Sequence.weighted_longest_common_subsequence, is_cut=True),
    }

    for v in _ret.values():
        v.return_more_info = True

    det_cut_text = nlp_utils.cut_word_by_jieba(det_text)
    gt_cut_text = nlp_utils.cut_word_by_jieba(gt_text)

    ret.update({k: v.f_measure(det_cut_text, gt_cut_text) for k, v in _ret.items()})

    df = pd.DataFrame(ret).T
    df = df.round(4)
    print(df)

    if save_path:
        os_lib.saver.auto_save(df, save_path)

    return df


def det_checkout_false_sample(gt_iter_data, det_iter_data, data_dir='checkout_data', image_dir=None, set_task=''):
    import numpy as np
    from utils import os_lib, visualize

    r = {}

    for ret in tqdm(gt_iter_data):
        r.setdefault(ret['_id'], {})['gt_boxes'] = ret['bboxes']
        r.setdefault(ret['_id'], {})['_id'] = ret['_id']

    for ret in tqdm(det_iter_data):
        r.setdefault(ret['_id'], {})['det_boxes'] = ret['bboxes']
        r.setdefault(ret['_id'], {})['confs'] = [1] * len(ret['bboxes'])

    gt_boxes = [v['gt_boxes'] for v in r.values()]
    det_boxes = [v['det_boxes'] for v in r.values()]
    confs = [v['confs'] for v in r.values()]
    _ids = [v['_id'] for v in r.values()]

    ret = object_detection.AP(return_more_info=True).mAP(gt_boxes, det_boxes, confs)
    r = ret['']

    image_dir = image_dir if image_dir is not None else f'{data_dir}/images'
    save_dir = f'{data_dir}/{set_task}'
    tp = r['tp']
    obj_idx = r['obj_idx']
    target_obj_idx = obj_idx[~tp]

    idx = np.unique(target_obj_idx)
    for i in idx:
        target_idx = obj_idx == i
        _tp = tp[target_idx]

        gt_box = gt_boxes[i]
        det_box = det_boxes[i]
        _id = _ids[i]
        image = os_lib.loader.load_img(f'{image_dir}/{_id}')

        false_obj_idx = np.where(~_tp)[0]

        gt_colors = [visualize.get_color_array(0) for _ in gt_box]
        det_colors = [visualize.get_color_array(0) for _ in det_box]
        for _ in false_obj_idx:
            det_colors[_] = visualize.get_color_array(len(ret) + 1)

        gt_image = visualize.ImageVisualize.box(image, gt_box, colors=gt_colors)
        det_image = visualize.ImageVisualize.box(image, det_box, colors=det_colors)

        image = np.concatenate([gt_image, det_image], axis=1)
        os_lib.saver.auto_save(image, f'{save_dir}/{_id}')

    return ret

