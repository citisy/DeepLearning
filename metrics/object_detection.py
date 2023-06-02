import numpy as np
from typing import List


class Area:
    @staticmethod
    def real_areas(box):
        """Area(box) = (x2 - x1) * (y2 - y1)

        Args:
            box(np.array): shape=(N, 4), 4 means xyxy.

        Returns:
            real_areas(np.array): shape=(N, )
        """
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    @staticmethod
    def intersection_areas(box1, box2):
        """Area(box1 & box2)

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            intersection_box(np.array): shape=(N, M)
        """
        # box1[:, None, 2:]: (N, 2) -> (N, 1, 2)
        # minimum((N, 1, 2), (M, 2)) -> broadcast ->  minimum((N, M, 2), (N, M, 2)) -> (N, M, 2)
        return (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)

    @classmethod
    def union_areas(cls, box1, box2, inter_area=None):
        """Area(box1 | box2)

        Arguments:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            union_areas(np.array): shape=(N, M)
        """
        area1 = cls.real_areas(box1)
        area2 = cls.real_areas(box2)
        if inter_area is None:
            inter_area = cls.intersection_areas(box1, box2)

        xv, yv = np.meshgrid(area1, area2)
        return xv.T + yv.T - inter_area

    @staticmethod
    def outer_areas(box1, box2):
        """outer rectangle area

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            outer_areas(np.array): shape=(N, M)
        """
        return (np.maximum(box1[:, None, 2:], box2[:, 2:]) - np.minimum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)


class Overlap:
    """only judge whether 2 object is overlap or not"""

    @staticmethod
    def line(line1, line2):
        """line1 = (a1, a2), line2 = (b1, b2),
        2 lines do not overlap means that
           a1 after b2 (a1 > b2)
        or a2 before b1 (a2 < b1)
        """
        a1, a2 = line1.T
        b1, b2 = line2.T

        f = (a1[:, None] > b2[None, :]) | (a2[:, None] < b1[None, :])

        return ~f

    @staticmethod
    def box(box1, box2):
        """box1 = (xa1, ya1, xa2, ya2), box2 = (xb1, yb1, xb2, yb2)
        2 boxes do not overlap means that
           point 'a1' in the right down of box2 (xa1 > xb2 | ya1 > yb2)
        or point 'a2' in the left top of box2 (xa2 < xb1 | ya2 < yb1)
        """

        xa1, ya1, xa2, ya2 = box1.T
        xb1, yb1, xb2, yb2 = box2.T

        f1 = (xa1[:, None] > xb2[None, :]) | (ya1[:, None] > yb2[None, :])
        f2 = (xa2[:, None] < xb1[None, :]) | (ya2[:, None] < yb1[None, :])

        return ~(f1 | f2)


class Iou:
    @staticmethod
    def iou(box1, box2, inter=None, union=None):
        """vanilla iou
        Area(box1 & box2) / Area(box1 | box2)
        See Also `torchvision.ops.box_iou`

        Arguments:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            iou_box(np.array): shape=(N, M)
        """
        box1, box2 = np.array(box1), np.array(box2)

        if inter is None:
            inter = Area.intersection_areas(box1, box2)

        if union is None:
            union = Area.union_areas(box1, box2, inter)

        return inter / union

    @staticmethod
    def siou(box1, box2):
        """single iou
        Area(box1 & box2) / Area(box2)

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            iou_box2(np.array): shape=(N, M)

        >>> a = np.array([[1, 1, 4, 4]])
        >>> b = np.array([[2, 2, 5, 5]])
        >>> Iou.siou(a, b) # ((4 - 2) * (4 - 2)) / (5 - 2) * (5 - 2)
        [[0.44444444]]
        """
        area2 = Area.real_areas(box2)  # (M,)
        inter = Area.intersection_areas(box1, box2)

        return inter / (area2 + 1E-12)  # iou = inter / (area2)

    @classmethod
    def giou(cls, box1, box2):
        """https://arxiv.org/pdf/1902.09630.pdf
        iou - (c - Area(box1 & box2)) / c
        See Also `torchvision.ops.generalized_box_iou`
        """
        outer = Area.outer_areas(box1, box2)
        inter = Area.intersection_areas(box1, box2)
        union = Area.union_areas(box1, box2)
        iou = cls.iou(box1, box2, inter=inter, union=union)

        return iou - (outer - union) / outer

    @classmethod
    def diou(cls, box1, box2, iou=None, eps=1e-7):
        """https://arxiv.org/pdf/1911.08287.pdf
        iou - d ^ 2 / c ^ 2
        See Also `torchvision.ops.distance_box_iou`
        """
        box1_center = (box1[:, 2:] - box1[:, :2]) / 2
        box2_center = (box2[:, 2:] - box2[:, :2]) / 2

        d = np.linalg.norm(box1_center[:, None, :] - box2_center, axis=2) ** 2
        c = np.linalg.norm((np.maximum(box1[:, None, 2:], box2[:, 2:]) - np.minimum(box1[:, None, :2], box2[:, :2])).clip(0), axis=2) ** 2 + eps

        if iou is None:
            iou = cls.iou(box1, box2)

        return iou - d / c

    @classmethod
    def ciou(cls, box1, box2, a=None, v=None, eps=1e-7):
        """https://arxiv.org/pdf/1911.08287.pdf
        diou - av
        See Also `torchvision.ops.complete_box_iou`
        """
        iou = cls.iou(box1, box2)
        diou = cls.diou(box1, box2, iou=iou, eps=eps)

        if v is None:
            box1_wh = box1[:, 2:] - box1[:, :2]
            box2_wh = box2[:, 2:] - box2[:, :2]
            b1 = np.arctan(box1_wh[:, 0] / box1_wh[:, 1])
            b2 = np.arctan(box2_wh[:, 0] / box2_wh[:, 1])

            v = 4 / np.pi ** 2 * ((b1[:, None] - b2) ** 2)

        if a is None:
            a = v / (1 - iou + v + eps)

        return diou - a * v


class ConfusionMatrix:
    def __init__(self, iou_method=None, **iou_method_kwarg):
        self.iou_method = iou_method or Iou.iou
        self.iou_method_kwarg = iou_method_kwarg

    def tp(self, gt_box, det_box, classes=None, iou_thres=0.5, iou=None):
        """

        Args:
            gt_box: (N, 4)
            det_box: (M, 4)
            classes (List[iter]):
            iou_thres:
            iou: (N, M)

        Returns:
            tp: (M, )
            iou: (N, M)

        """
        if iou is None:
            if classes is not None:
                true_class, pred_class = classes
                offset = np.max(np.concatenate([gt_box, det_box], 0))
                gt_box += (true_class * offset)[:, None]
                det_box += (pred_class * offset)[:, None]

            iou = self.iou_method(gt_box, det_box, **self.iou_method_kwarg)

        tp = np.zeros((det_box.shape[0],), dtype=bool)
        idx = np.where((iou >= iou_thres))
        idx = np.stack(idx, 1)
        if idx.shape[0]:
            matches = np.concatenate((idx, iou[idx[:, 0], idx[:, 1]][:, None]), 1)  # [idx of gt, idx of det, iou]
            if idx.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp[matches[:, 1].astype(int)] = True

        # fast
        # tp = np.any(iou >= iou_thres, axis=0, dtype=bool)

        return tp, iou

    def fp(self, *args, **kwargs):
        """see also `ConfusionMatrix.tp`"""
        tp, iou = self.tp(*args, **kwargs)
        return -tp, iou


class PR:
    def __init__(self, iou_method=None, **iou_method_kwarg):
        self.confusion_matrix = ConfusionMatrix(iou_method, **iou_method_kwarg)

    def recall(self, gt_box=None, det_box=None, conf=None, tp=None, n_true=None, eps=1e-16, iou_thres=0.5):
        """

        Args:
            gt_box:
            det_box:
            conf:
            tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            n_true:
            iou_thres:

        Returns:

        """
        sort_idx = None
        if tp is None:
            sort_idx = np.argsort(-conf)
            det_box = det_box[sort_idx]
            tp, iou = self.confusion_matrix.tp(gt_box, det_box, iou_thres=iou_thres)
            n_true = len(gt_box)

        acc_tp = np.cumsum(tp)

        return acc_tp / (n_true + eps), tp, sort_idx

    def precision(self, gt_box=None, det_box=None, conf=None, tp=None, eps=1e-16, iou_thres=0.5):
        """

        Args:
            gt_box:
            det_box:
            conf:
            tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            iou_thres:

        Returns:

        """
        sort_idx = None
        if tp is None:
            sort_idx = np.argsort(-conf)
            det_box = det_box[sort_idx]
            tp, iou = self.confusion_matrix.tp(gt_box, det_box, iou_thres=iou_thres)

        acc_tp = np.cumsum(tp)
        n = np.cumsum(np.ones_like(acc_tp))

        return acc_tp / (n + eps), tp, sort_idx


class AP:
    """https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/paper_survey_on_performance_metrics_for_object_detection_algorithms.pdf"""

    def __init__(
            self, ap_method=None, ap_method_kwargs=dict(),
            iou_method=None, iou_method_kwarg=dict(),
            return_more_info=False
    ):
        self.confusion_matrix = ConfusionMatrix(iou_method, **iou_method_kwarg)
        self.pr = PR(iou_method, **iou_method_kwarg)
        self.ap_method = ap_method or self.continuous
        self.ap_method_kwargs = ap_method_kwargs
        self.return_more_info = return_more_info

    @staticmethod
    def interp(mean_recall, mean_precision, point=11, **kwargs):
        x = np.linspace(0, 1, point)
        ap = np.trapz(np.interp(x, mean_recall, mean_precision), x)  # integrate

        return ap

    @staticmethod
    def continuous(mean_recall, mean_precision, **kwargs):
        i = np.where(mean_recall[1:] != mean_recall[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mean_recall[i + 1] - mean_recall[i]) * mean_precision[i + 1])  # area under curve

        return ap

    def per_class_ap(
            self, gt_box=None, det_box=None, conf=None,
            tp=None, n_true=None, iou_thres=0.5
    ):
        """ap computation core method, ap of per objection per class

        Args:
            gt_box (np.ndarray):
            det_box (np.ndarray):
            conf (np.ndarray):
            tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            iou_thres:
            n_true (int): if not set, `gt_box` is necessary

        Returns:

        """

        recall, tp, sort_idx = self.pr.recall(gt_box, det_box, conf, tp, n_true=n_true, iou_thres=iou_thres)
        precision, tp, _ = self.pr.precision(gt_box, det_box, conf, tp, iou_thres=iou_thres)

        # Append sentinel values to beginning and end
        mean_recall = np.concatenate(([0.0], recall, [1.0]))
        mean_precision = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mean_precision = np.flip(np.maximum.accumulate(np.flip(mean_precision)))

        if tp.size:
            ap = self.ap_method(mean_recall, mean_precision, **self.ap_method_kwargs)
        else:
            ap = 0

        true_positive = int(np.sum(tp))
        false_positive = len(tp) - true_positive

        r = true_positive / n_true
        p = true_positive / len(tp)
        f1 = 2 * p * r / (p + r + 1e-16)

        ret = {
            f'ap': round(ap, 6),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'n_true': n_true,
            'p': round(p, 6),
            'r': round(r, 6),
            'f1': round(f1, 6)
        }

        if self.return_more_info:
            ret.update(
                tp=tp,
                sort_idx=sort_idx,
                recall=recall,
                precision=precision,
                mean_recall=mean_recall,
                mean_precision=mean_precision
            )

        return ret

    def ap_thres(
            self, gt_box=None, det_box=None, conf=None, classes=None,
            tp=None, n_true=None, iou_thres=0.5
    ):
        """AP@iou_thres for each objection

        Args:
            gt_box (np.ndarray):
            det_box (np.ndarray):
            conf (np.ndarray):
            classes (List[np.ndarray]):
            tp: if set, `gt_box`, `det_box` is not necessary
            n_true (int or List[int]): if not set, `gt_box` is necessary
            iou_thres:

        Returns:
        """
        if classes is None:
            return self.per_class_ap(gt_box, det_box, conf, tp, n_true=n_true, iou_thres=iou_thres)
        else:
            unique_class = np.unique(np.concatenate(classes))
            true_class, pred_class = classes

            if tp is None:
                tp, iou = self.confusion_matrix.tp(gt_box, det_box, classes, iou_thres=iou_thres)

            sort_idx = np.argsort(-conf)
            tmp_tp = tp[sort_idx]
            tmp_pred_class = pred_class[sort_idx]

            results = {}

            for i, c in enumerate(unique_class):
                _n_true = len(gt_box[true_class == c]) if gt_box is not None else n_true[i]
                r = self.per_class_ap(
                    tp=tmp_tp[tmp_pred_class == c],
                    n_true=_n_true,
                    iou_thres=iou_thres
                )

                if self.return_more_info:
                    r.update(
                        sort_idx=sort_idx[pred_class == c],
                        true_class=true_class[true_class == c],
                        pred_class=pred_class[pred_class == c],
                        det_box=det_box[pred_class == c] if det_box is not None else None,
                        gt_box=gt_box[true_class == c] if det_box is not None else None,
                        conf=conf[pred_class == c]
                    )

                results[c] = r

            return results

    def mAP(self, gt_boxes, det_boxes, confs, classes=None, iou_thres=0.5):
        """AP@iou_thres for all objection

        Args:
            gt_boxes (List[np.ndarray]):
            det_boxes (List[np.ndarray]):
            confs (List[np.ndarray]):
            classes (List[List[np.ndarray]]):
            iou_thres:

        Returns:

        """
        if classes is None:
            tps = []
            for data in zip(gt_boxes, det_boxes):
                tp, iou = self.confusion_matrix.tp(*data, iou_thres=iou_thres)

                tps.append(tp)

            tps = np.concatenate(tps, axis=0)

        else:
            tmp_true_classes, tmp_pred_classes = [], []
            tps = []
            for g, d, tc, pc in zip(gt_boxes, det_boxes, *classes):
                tp, iou = self.confusion_matrix.tp(g, d, [tc, pc], iou_thres=iou_thres)
                tps.append(tp)
                tmp_true_classes.append(tc)
                tmp_pred_classes.append(pc)

            tps = np.concatenate(tps, axis=0)
            classes = [np.concatenate(tmp_true_classes), np.concatenate(tmp_pred_classes)]

        det_boxes = np.concatenate(det_boxes, axis=0)
        gt_boxes = np.concatenate(gt_boxes, axis=0)
        confs = np.concatenate(confs, axis=0)

        return self.ap_thres(gt_boxes, det_boxes, confs, classes=classes, tp=tps, iou_thres=iou_thres)

    def ap_thres_range(
            self, gt_box, det_box, conf, classes=None,
            tps=None, n_true=None, thres_range=np.arange(0.5, 1, 0.05)
    ):
        """AP@thres_range for each objection

        Args:
            gt_box (np.ndarray):
            det_box (np.ndarray):
            conf (np.ndarray):
            classes (List[np.ndarray]):
            tps (List[np.ndarray]): if set, `gt_box`, `det_box` is not necessary
            n_true (int or List[int]): if not set, `gt_box` is necessary
            thres_range (iterator)

        """
        _ret = {}

        # note that, it is slower, 'cause computing iou matrix each iou threshold,
        # but it is easy to complete in this way, and I have no plan to refactor the code
        for i, thres in enumerate(thres_range):
            tp = tps[i] if tps is not None else None
            result = self.ap_thres(gt_box, det_box, conf, classes, tp=tp, n_true=n_true, iou_thres=thres)

            for k, v in result.items():
                tmp = _ret.setdefault(k, {})
                for kk, vv in v.items():
                    tmp.setdefault(kk, []).append(vv)

        ret = {k: {} for k in _ret}
        for k, v in _ret.items():
            ret[k][f'ap@{thres_range[0]:.2f}:{thres_range[-1]:.2f}'] = np.mean(v['ap'])

            if self.return_more_info:
                ret[k].update(v)
            else:
                ret[k].update({kk: vv[0] for kk, vv in v.items()})

        return ret

    def mAP_thres_range(
            self, gt_boxes, det_boxes, confs, classes=None,
            thres_range=np.arange(0.5, 1, 0.05),
    ):
        """AP@thres_range for all objection

        Args:
            gt_boxes (List[np.ndarray]):
            det_boxes (List[np.ndarray]):
            confs (List[np.ndarray]):
            classes (List[List[np.ndarray]]):
            thres_range:

        Returns:

        """
        _tps = []
        tmp_iou = [None for _ in range(len(gt_boxes))]

        for thres in thres_range:
            if classes is None:
                tps = []
                for i, data in enumerate(zip(gt_boxes, det_boxes)):
                    tp, tmp_iou[i] = self.confusion_matrix.tp(*data, iou=tmp_iou[i], iou_thres=thres)
                    tps.append(tp)

                tps = np.concatenate(tps, axis=0)
                _tps.append(tps)

            else:
                tps = []
                for i, (g, d, tc, pc) in enumerate(zip(gt_boxes, det_boxes, *classes)):
                    tp, tmp_iou[i] = self.confusion_matrix.tp(g, d, [tc, pc], iou=tmp_iou[i], iou_thres=thres)
                    tps.append(tp)

                tps = np.concatenate(tps, axis=0)
                _tps.append(tps)

        det_boxes = np.concatenate(det_boxes, axis=0)
        gt_boxes = np.concatenate(gt_boxes, axis=0)
        confs = np.concatenate(confs, axis=0)

        if classes is not None:
            tmp_true_classes, tmp_pred_classes = [], []
            for tc, pc in zip(*classes):
                tmp_true_classes.append(tc)
                tmp_pred_classes.append(pc)

            classes = [np.concatenate(tmp_true_classes), np.concatenate(tmp_pred_classes)]

        return self.ap_thres_range(gt_boxes, det_boxes, confs, classes=classes, tps=_tps, thres_range=thres_range)


confusion_matrix = ConfusionMatrix()
pr = PR()
ap = AP()
