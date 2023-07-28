import time
import numpy as np
from typing import List, Iterable, Iterator
from tqdm import tqdm


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
    def union_areas(cls, box1, box2, inter=None):
        """Area(box1 | box2)

        Arguments:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            union_areas(np.array): shape=(N, M)
        """
        area1 = cls.real_areas(box1)
        area2 = cls.real_areas(box2)
        if inter is None:
            inter = cls.intersection_areas(box1, box2)

        xv, yv = np.meshgrid(area1, area2)
        return xv.T + yv.T - inter

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

    @staticmethod
    def intersection_areas1D(box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        return (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    @classmethod
    def union_areas1D(cls, box1, box2, inter=None):
        area1 = cls.real_areas(box1)
        area2 = cls.real_areas(box2)
        if inter is None:
            inter = cls.intersection_areas1D(box1, box2)

        return area1 + area2 - inter

    @staticmethod
    def outer_areas1D(box1, box2):
        """outer rectangle area

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            outer_areas(np.array): shape=(N, M)
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
        return (np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)).clip(0) * (np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)).clip(0)


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
            iou_mat(np.array): shape=(N, M)
        """
        box1, box2 = np.array(box1), np.array(box2)

        if inter is None:
            inter = Area.intersection_areas(box1, box2)

        if union is None:
            union = Area.union_areas(box1, box2, inter)

        return inter / union

    @staticmethod
    def siou(box1, box2, inter=None):
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
        if inter is None:
            inter = Area.intersection_areas(box1, box2)

        return inter / (area2 + 1E-12)  # iou = inter / (area2)

    @classmethod
    def miou(cls, box1, box2, inter=None):
        """mean iou
        (siou(box1, box2) + siou(box2, box1).T) / 2

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            iou_box(np.array): shape=(N, M)

        """
        box1, box2 = np.array(box1), np.array(box2)

        if inter is None:
            inter = Area.intersection_areas(box1, box2)

        siou1 = cls.siou(box1, box2, inter=inter)
        siou2 = cls.siou(box2, box1, inter=inter.T).T

        return (siou1 + siou2) / 2

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

    @staticmethod
    def iou1D(box1, box2, inter=None, union=None):
        """box1 and box2 must have the same shape

        Args:
            box1: (N, 4)
            box2: (N, 4)
            inter:
            union:

        Returns:
            iou_mat(np.array): shape=(N, )
        """

        if inter is None:
            inter = Area.intersection_areas1D(box1, box2)

        if union is None:
            union = Area.union_areas1D(box1, box2, inter)

        return inter / union

    @staticmethod
    def siou1D(box1, box2):
        area2 = Area.real_areas(box2)
        inter = Area.intersection_areas1D(box1, box2)

        return inter / (area2 + 1E-12)

    @classmethod
    def giou1D(cls, box1, box2):
        outer = Area.outer_areas1D(box1, box2)
        inter = Area.intersection_areas1D(box1, box2)
        union = Area.union_areas1D(box1, box2)
        iou = cls.iou1D(box1, box2, inter=inter, union=union)

        return iou - (outer - union) / outer

    @classmethod
    def diou1D(cls, box1, box2, iou=None, eps=1e-7):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        c = (np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)) ** 2 + (np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)) ** 2 + eps
        d = ((b2_x1 - b1_x1 + b2_x2 - b1_x2) ** 2 + (b2_y1 - b1_y1 + b2_y2 - b1_y2) ** 2) / 4

        if iou is None:
            iou = cls.iou1D(box1, box2)

        return iou - d / c

    @classmethod
    def ciou1D(cls, box1, box2, a=None, v=None, eps=1e-7):
        iou = cls.iou1D(box1, box2)
        diou = cls.diou1D(box1, box2, iou=iou, eps=eps)

        if v is None:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

            v = (4 / np.pi ** 2) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2)

        if a is None:
            a = v / (1 - iou + v + eps)

        return diou - a * v


class ConfusionMatrix:
    def __init__(self, iou_method=None, **iou_method_kwarg):
        self.iou_method = iou_method or Iou.iou
        self.iou_method_kwarg = iou_method_kwarg

    def tp(self, gt_box, det_box, conf=None, _class=None, iou=None, iou_thres=0.5):
        """

        Args:
            gt_box: (N, 4)
            det_box: (M, 4)
            conf:
            _class (List[iter]):
            iou_thres:
            iou: (N, M)

        Returns:
            tp: (M, )
            iou: (N, M)

        """

        sort_idx = None
        if conf is not None:
            conf = np.array(conf)
            sort_idx = np.argsort(-conf)
            det_box = det_box[sort_idx]

        if iou is None:
            if _class is not None:
                true_class, pred_class = _class
                if sort_idx is not None:
                    pred_class = pred_class[sort_idx]
                offset = np.max(np.concatenate([gt_box, det_box], 0))
                gt_box = gt_box + (true_class * offset)[:, None]
                det_box = det_box + (pred_class * offset)[:, None]

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

        if sort_idx is not None:
            sort_idx = np.argsort(sort_idx)
            tp = tp[sort_idx]
            iou = iou[:, sort_idx]

        return dict(
            tp=tp,
            acc_tp=np.sum(tp),
            iou=iou,  # have been sorted by conf
        )

    def cp(self, gt_box):
        return dict(
            cp=[True] * len(gt_box),
            acc_cp=len(gt_box)
        )

    def op(self, det_box):
        return dict(
            op=[True] * len(det_box),
            acc_op=len(det_box)
        )


class PR:
    def __init__(self, eps=1e-16, iou_method=None, **iou_method_kwarg):
        self.eps = eps
        self.confusion_matrix = ConfusionMatrix(iou_method, **iou_method_kwarg)

        # alias functions
        self.recall = self.tpr
        self.precision = self.ppv

    def get_pr(self, gt_boxes=None, det_boxes=None, confs=None, classes=None, ious=None, iou_thres=0.5):
        rs = {}
        gt_obj_idx, det_obj_idx = [], []
        for i, (g, d) in enumerate(zip(gt_boxes, det_boxes)):
            conf = confs[i] if confs is not None else None
            _class = [classes[0][i], classes[1][i]] if classes is not None else None
            iou = ious[i] if ious is not None else None
            r = self.confusion_matrix.tp(g, d, conf, _class, iou=iou, iou_thres=iou_thres)
            for k, v in r.items():
                rs.setdefault(k, []).append(v)

            r = self.confusion_matrix.cp(g)
            for k, v in r.items():
                rs.setdefault(k, []).append(v)

            r = self.confusion_matrix.op(d)
            for k, v in r.items():
                rs.setdefault(k, []).append(v)

            gt_obj_idx.append([i] * len(g))
            det_obj_idx.append([i] * len(d))

        tp = np.concatenate(rs['tp'])
        cp = np.concatenate(rs['cp'])
        op = np.concatenate(rs['op'])
        gt_idx = np.arange(len(cp), dtype=int)
        det_idx = np.arange(len(op), dtype=int)
        gt_obj_idx = np.concatenate(gt_obj_idx)
        det_obj_idx = np.concatenate(det_obj_idx)

        results = {}
        if classes is None:
            acc_tp = rs['acc_tp']
            acc_cp = rs['acc_cp']
            acc_op = rs['acc_op']

            ret = dict(
                tp=tp,
                cp=cp,
                op=op,
                gt_idx=gt_idx,
                det_idx=det_idx,
                gt_obj_idx=gt_obj_idx,
                det_obj_idx=det_obj_idx
            )
            ret.update(self.tpr(acc_tp=sum(acc_tp), acc_cp=sum(acc_cp), iou_thres=iou_thres))
            ret.update(self.ppv(acc_tp=sum(acc_tp), acc_op=sum(acc_op), iou_thres=iou_thres))
            results[''] = ret

        else:
            tmp_gt_classes, tmp_det_classes = [], []
            for i, (gc, dc) in enumerate(zip(*classes)):
                tmp_gt_classes.append(gc)
                tmp_det_classes.append(dc)

            gt_class, det_class = np.concatenate(tmp_gt_classes), np.concatenate(tmp_det_classes)
            unique_class = np.unique(np.concatenate([gt_class, det_class]))

            for i, c in enumerate(unique_class):
                gt_cls_idx = gt_class == c
                det_cls_idx = det_class == c

                cls_tp = tp[det_cls_idx]
                cls_cp = cp[gt_cls_idx]
                cls_op = op[det_cls_idx]

                ret = {}
                ret.update(self.tpr(acc_tp=sum(cls_tp), acc_cp=sum(cls_cp), iou_thres=iou_thres))
                ret.update(self.ppv(acc_tp=sum(cls_tp), acc_op=sum(cls_op), iou_thres=iou_thres))

                ret.update(
                    tp=cls_tp,
                    cp=cls_cp,
                    op=cls_op,
                    gt_idx=gt_idx[gt_cls_idx],
                    det_idx=det_idx[det_cls_idx],
                    gt_obj_idx=gt_obj_idx[gt_cls_idx],
                    det_obj_idx=det_obj_idx[det_cls_idx]
                )

                results[c] = ret

        return results, rs['iou']

    def tpr(self, gt_box=None, det_box=None, conf=None, _class=None, acc_tp=None, acc_cp=None, iou_thres=0.5):
        """

        Args:
            gt_box:
            det_box:
            conf:
            _class:
            acc_tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            acc_cp: if set, `gt_box`, not necessary
            iou_thres:

        Returns:

        """
        r = {}

        if acc_tp is None:
            r.update(self.confusion_matrix.tp(gt_box, det_box, conf, _class=_class, iou_thres=iou_thres))
            acc_tp = r.pop('acc_tp')

        if acc_cp is None:
            r.update(self.confusion_matrix.cp(gt_box))
            acc_cp = r.pop('acc_cp')

        ret = dict(
            r=acc_tp / (acc_cp + self.eps),
            acc_tp=acc_tp,
            acc_cp=acc_cp,
            **r
        )

        return ret

    def ppv(self, gt_box=None, det_box=None, conf=None, _class=None, acc_tp=None, acc_op=None, iou_thres=0.5):
        """

        Args:
            gt_box:
            det_box:
            conf:
            _class:
            acc_tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            acc_op: if set, `det_box`, not necessary
            iou_thres:

        Returns:

        """
        r = {}

        if acc_tp is None:
            r.update(self.confusion_matrix.tp(gt_box, det_box, conf, _class=_class, iou_thres=iou_thres))
            acc_tp = r.pop('acc_tp')

        if acc_op is None:
            r.update(self.confusion_matrix.op(det_box))
            acc_op = r.pop('acc_op')

        ret = dict(
            p=acc_tp / (acc_op + self.eps),
            acc_tp=acc_tp,
            acc_op=acc_op,
            **r
        )

        return ret


class ApMethod:
    @staticmethod
    def interp(mean_recall, mean_precision, point=101, **kwargs):
        x = np.linspace(0, 1, point)
        return np.trapz(np.interp(x, mean_recall, mean_precision), x)  # integrate

    @staticmethod
    def continuous(mean_recall, mean_precision, **kwargs):
        i = np.where(mean_recall[1:] != mean_recall[:-1])[0]  # points where x axis (recall) changes
        return np.sum((mean_recall[i + 1] - mean_recall[i]) * mean_precision[i + 1])  # area under curve


class AP:
    """https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/paper_survey_on_performance_metrics_for_object_detection_algorithms.pdf"""

    def __init__(
            self, return_more_info=False, eps=1e-6, ap_method=None, ap_method_kwargs=dict(),
            pr_method=None, **pr_method_kwarg
    ):
        self.pr = pr_method(**pr_method_kwarg) if pr_method is not None else PR(**pr_method_kwarg)
        self.ap_method = ap_method or ApMethod.interp
        self.ap_method_kwargs = ap_method_kwargs
        self.return_more_info = return_more_info
        self.eps = eps

    def mAP_thres(self, gt_boxes, det_boxes, confs=None, classes=None, iou_thres=0.5):
        """AP@iou_thres for all objection

        Args:
            gt_boxes (List[np.ndarray]):
            det_boxes (List[np.ndarray]):
            confs (List[np.ndarray]):
            classes (List[List[np.ndarray]]):
            iou_thres:

        Returns:

        Usage:
            See Also `quick_metric()` or `shortlist_false_sample()`
        """
        assert len(gt_boxes)
        assert len(det_boxes)

        _results, _ = self.pr.get_pr(gt_boxes, det_boxes, confs, classes, iou_thres=iou_thres)
        return self.ap_thres(_results, confs)

    def mAP_thres_range(self, gt_boxes, det_boxes, confs, classes=None, thres_range=np.arange(0.5, 1, 0.05)):
        """AP@thres_range for all objection

        Args:
            gt_boxes (List[np.ndarray]):
            det_boxes (List[np.ndarray]):
            confs (List[np.ndarray]):
            classes (List[List[np.ndarray]]):
            thres_range:

        Returns:

        Usage:
            See Also `AP.mAP`
        """
        ious = None
        _ret = {}

        for iou_thres in thres_range:
            _results, ious = self.pr.get_pr(gt_boxes, det_boxes, confs, classes, ious=ious, iou_thres=iou_thres)
            results = self.ap_thres(_results, confs)

            for k, v in results.items():
                tmp = _ret.setdefault(k, {})
                for kk, vv in v.items():
                    tmp.setdefault(kk, []).append(vv)

        ret = {k: {} for k in _ret}
        for k, v in _ret.items():
            ret[k][f'ap@{thres_range[0]:.2f}:{thres_range[-1]:.2f}'] = np.mean(v['ap'])

            if self.return_more_info:
                ret[k].update(v)
            else:
                ret[k].update({kk: vv[0] for kk, vv in v.items()})  # only return results of ap@0.5

        return ret

    def ap_thres(self, _results, confs=None):
        results = {}

        if confs is not None:
            confs = np.concatenate(confs)

        for k, _r in _results.items():
            r = _r.pop('r')
            p = _r.pop('p')
            tp = _r['tp']
            op = _r['op']
            acc_tp = _r.pop('acc_tp')
            acc_cp = _r.pop('acc_cp')
            acc_op = _r.pop('acc_op')
            f = 2 * p * r / (p + r + self.eps)

            if confs is not None:
                det_idx = _r['det_idx']
                _confs = confs[det_idx]
                sort_idx = np.argsort(-_confs)
                tp = tp[sort_idx]

            # accumulate the recall and precision, the last one is the total recall and precision
            cumsum_tp = np.cumsum(tp)
            acc_recall = cumsum_tp / (acc_cp + self.eps)
            acc_precision = cumsum_tp / np.cumsum(op)

            # count ap
            # Append sentinel values to beginning and end
            mean_recall = np.concatenate(([0.0], acc_recall, [1.0]))
            mean_precision = np.concatenate(([1.0], acc_precision, [0.0]))

            # Compute the precision envelope
            mean_precision = np.flip(np.maximum.accumulate(np.flip(mean_precision)))

            if tp.size:
                ap = self.ap_method(mean_recall, mean_precision, **self.ap_method_kwargs)
            else:
                ap = 0

            ret = dict(
                ap=ap,
                p=round(p, 6),
                r=round(r, 6),
                f=round(f, 6),
                true_positive=acc_tp,
                false_positive=acc_op - acc_tp,
                n_true=acc_cp,
                n_pred=acc_op
            )

            if self.return_more_info:
                ret.update(
                    **_r
                )

            results[k] = ret

        return results


class EasyMetric:
    def __init__(self, iou_thres=0.5, cls_alias=None, verbose=True, stdout_method=print, **ap_kwargs):
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.stdout_method = stdout_method
        self.ap = AP(**ap_kwargs)

        if isinstance(cls_alias, list):
            cls_alias = {i: c for i, c in enumerate(cls_alias)}

        self.cls_alias = cls_alias

    def get_rets(self, gt_iter_data, det_iter_data, image_dir=None):
        rets = {}
        for ret in tqdm(gt_iter_data, desc='load gt data'):
            dic = rets.setdefault(ret['_id'], {})
            dic['gt_boxes'] = ret['bboxes']
            dic['gt_classes'] = ret['classes']
            dic['image_dir'] = ret['image_dir'] if 'image_dir' in ret else image_dir
            if 'confs' in ret:
                dic['gt_confs'] = ret['confs']

        for ret in tqdm(det_iter_data, desc='load det data'):
            dic = rets[ret['_id']]
            dic['det_boxes'] = ret['bboxes']
            dic['det_classes'] = ret['classes']
            dic['confs'] = ret['confs']

        gt_boxes = [v['gt_boxes'] for v in rets.values()]
        det_boxes = [v['det_boxes'] for v in rets.values()]
        confs = [v['confs'] for v in rets.values()]
        gt_classes = [v['gt_classes'] for v in rets.values()]
        det_classes = [v['det_classes'] for v in rets.values()]
        _ids = list(rets.keys())

        return rets, _ids, gt_boxes, det_boxes, gt_classes, det_classes, confs

    def quick_metric(self, gt_iter_data, det_iter_data, is_mAP=True, save_path=None):
        """

        Args:
            gt_iter_data (Iterable):
            det_iter_data (Iterable):
            is_mAP (bool):
            save_path (str):

        Usage:
            .. code-block:: python

                # use yolov5 type data result to metric
                from cv_data_parse.YoloV5 import Loader
                data_dir = 'your data dir'
                sub_dir = 'model version, e.g. backbone, etc.'
                task = 'strategy version, e.g. dataset, train datetime, apply params, etc.'

                loader = Loader(data_dir)
                gt_iter_data = loader.load_full_labels(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full_labels(sub_dir=sub_dir, task=task)

                EasyMetric().quick_metric(gt_iter_data, det_iter_data)

        """
        import pandas as pd

        rets, _ids, gt_boxes, det_boxes, gt_classes, det_classes, confs = self.get_rets(gt_iter_data, det_iter_data)

        if is_mAP:
            ret = self.ap.mAP_thres(gt_boxes, det_boxes, confs, classes=[gt_classes, det_classes], iou_thres=self.iou_thres)
        else:
            ret = self.ap.mAP_thres_range(gt_boxes, det_boxes, confs, classes=[gt_classes, det_classes], thres_range=np.arange(self.iou_thres, 1, 0.05))

        cls_alias = self.cls_alias or {k: k for k in ret}
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        df = pd.DataFrame(ret).T
        df.index = [cls_alias[i] for i in df.index]

        s = df.sum(axis=0)
        m = df.mean(axis=0)
        df.loc['sum'] = s
        df.loc['mean'] = m
        df = df.round(6)

        if self.verbose:
            self.stdout_method(df)

        if save_path:
            from utils import os_lib
            os_lib.Saver(verbose=self.verbose, stdout_method=self.stdout_method).auto_save(df, save_path)

        return df

    def checkout_false_sample(self, gt_iter_data, det_iter_data, data_dir='checkout_data', image_dir=None, save_res_dir=None):
        """

        Args:
            gt_iter_data (Iterable):
            det_iter_data (Iterable):
            data_dir (str):
            image_dir (str):
                which dir can get the image, `image_path = f'{image_dir}/{_id}'`
                if gt_iter_data set the key of image_dir, use the value
                else, if image_dir is set, use it, else use f'{data_dir}/images'

            save_res_dir:

        Usage:
            .. code-block:: python

                # use yolov5 type data result to metric
                from cv_data_parse.cv_data_parse.YoloV5 import Loader

                data_dir = 'your data dir'
                sub_dir = 'model version, e.g. backbone, etc.'
                task = 'strategy version, e.g. dataset, train time, apply params, etc.'
                set_task = f'{sub_dir}/{task}'
                image_dir = f'{data_dir}/images/{task}'
                save_res_dir = f'{data_dir}/visuals/false_samples/{set_task}'

                loader = Loader(data_dir)
                gt_iter_data = loader.load_full_labels(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full_labels(sub_dir=sub_dir, task=task)
                cls_alias = {}  # set if you want to convert class name

                EasyMetric(cls_alias=cls_alias).checkout_false_sample(gt_iter_data, det_iter_data, data_dir=data_dir, image_dir=image_dir, save_res_dir=save_res_dir)

        """
        from utils import os_lib, visualize
        from data_parse.cv_data_parse.base import DataVisualizer

        image_dir = image_dir if image_dir is not None else f'{data_dir}/images'
        save_res_dir = save_res_dir if save_res_dir is not None else f'{data_dir}/visuals/false_samples'

        rets, _ids, gt_boxes, det_boxes, gt_classes, det_classes, confs = self.get_rets(gt_iter_data, det_iter_data, image_dir=image_dir)
        self.ap.return_more_info = True
        ret = self.ap.mAP_thres(gt_boxes, det_boxes, confs, classes=[gt_classes, det_classes], iou_thres=self.iou_thres)

        cls_alias = self.cls_alias or {k: k for k in ret}

        for cls, r in ret.items():
            save_dir = f'{save_res_dir}/{cls_alias[cls]}'
            tp = r['tp']
            det_obj_idx = r['det_obj_idx']
            target_obj_idx = det_obj_idx[~tp]

            idx = np.unique(target_obj_idx)
            for i in idx:
                target_idx = det_obj_idx == i
                _tp = tp[target_idx]

                gt_class = gt_classes[i]
                gt_box = gt_boxes[i]
                conf = confs[i]
                det_class = det_classes[i]
                det_box = det_boxes[i]
                _id = _ids[i]

                image = os_lib.loader.load_img(f'{rets[_id]["image_dir"]}/{_id}')

                false_obj_idx = np.where(det_class == cls)[0]
                false_obj_idx = false_obj_idx[~_tp]

                gt_class = [int(c) for c in gt_class]
                det_class = [int(c) for c in det_class]

                cls_alias[-1] = cls_alias[cls]

                for _ in false_obj_idx:
                    det_class[_] = -1

                tmp_gt = [dict(_id=_id, image=image, bboxes=gt_box, classes=gt_class)]
                tmp_det = [dict(image=image, bboxes=det_box, classes=det_class, confs=conf)]
                visualizer = DataVisualizer(save_dir, verbose=self.verbose, stdout_method=self.stdout_method)
                visualizer(tmp_gt, tmp_det, cls_alias=cls_alias)

        return ret

    def checkout_diff_sample(self, gt_iter_data, det_iter_data, data_dir='checkout_data', image_dir=None, save_res_dir=None):
        """

        Args:
            gt_iter_data (Iterable):
            det_iter_data (Iterable):
            data_dir (str):
            image_dir (str):
                which dir can get the image, `image_path = f'{image_dir}/{_id}'`
                if gt_iter_data set the key of image_dir, use the value
                else, if image_dir is set, use it, else use f'{data_dir}/images'

            save_res_dir:

        Usage:
            .. code-block:: python

                # use yolov5 type data result to metric
                from cv_data_parse.cv_data_parse.YoloV5 import Loader

                data_dir = 'your data dir'
                sub_dir = 'model version, e.g. backbone, etc.'
                task = 'strategy version, e.g. dataset, train time, apply params, etc.'
                set_task = f'{sub_dir}/{task}'
                image_dir = f'{data_dir}/images/{task}'
                save_res_dir = f'{data_dir}/visuals/diff_samples/{set_task}'

                loader = Loader(data_dir)
                gt_iter_data = loader.load_full_labels(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full_labels(sub_dir=sub_dir, task=task)
                cls_alias = {}  # set if you want to convert class name

                EasyMetric(cls_alias=cls_alias).checkout_diff_sample(gt_iter_data, det_iter_data, data_dir=data_dir, image_dir=image_dir, save_res_dir=save_res_dir)

        """

        from utils import os_lib, visualize
        from data_parse.cv_data_parse.base import DataVisualizer

        image_dir = image_dir if image_dir is not None else f'{data_dir}/images'
        save_res_dir = save_res_dir if save_res_dir is not None else f'{data_dir}/visuals/diff_samples'

        rets, _ids, gt_boxes, det_boxes, gt_classes, det_classes, confs = self.get_rets(gt_iter_data, det_iter_data, image_dir=image_dir)
        gt_confs = [v['gt_confs'] for v in rets.values()]
        self.ap.return_more_info = True
        ret1 = self.ap.mAP_thres(gt_boxes, det_boxes, confs, classes=[gt_classes, det_classes], iou_thres=self.iou_thres)
        ret2 = self.ap.mAP_thres(det_boxes, gt_boxes, gt_confs, classes=[det_classes, gt_classes], iou_thres=self.iou_thres)

        cls_alias = self.cls_alias or {k: k for k in ret1}

        tmp_ret = {}

        for cls, r in ret1.items():
            tp = r['tp']
            det_obj_idx = r['det_obj_idx']
            target_obj_idx = det_obj_idx[~tp]

            idx = np.unique(target_obj_idx)
            for i in idx:
                target_idx = det_obj_idx == i
                _id = _ids[i]
                _tp = tp[target_idx]

                tmp_ret.setdefault((_id, cls, i), {})['det_tp'] = _tp

        for cls, r in ret2.items():
            tp = r['tp']
            det_obj_idx = r['det_obj_idx']
            target_obj_idx = det_obj_idx[~tp]

            idx = np.unique(target_obj_idx)
            for i in idx:
                target_idx = det_obj_idx == i
                _id = _ids[i]
                _tp = tp[target_idx]

                tmp_ret.setdefault((_id, cls, i), {})['gt_tp'] = _tp

        for (_id, cls, i), dic in tmp_ret.items():
            save_dir = f'{save_res_dir}/{cls_alias[cls]}'
            gt_class = gt_classes[i]
            gt_box = gt_boxes[i]
            gt_conf = gt_confs[i]
            det_conf = confs[i]
            det_class = det_classes[i]
            det_box = det_boxes[i]

            image = os_lib.loader.load_img(f'{rets[_id]["image_dir"]}/{_id}')

            gt_class = [int(c) for c in gt_class]
            det_class = [int(c) for c in det_class]

            cls_alias[-1] = cls_alias[cls]

            if 'det_tp' in dic:
                _tp = dic['det_tp']
                false_obj_idx = np.where(det_class == cls)[0]
                false_obj_idx = false_obj_idx[~_tp]
                for _ in false_obj_idx:
                    det_class[_] = -1

            if 'gt_tp' in dic:
                _tp = dic['gt_tp']
                false_obj_idx = np.where(gt_class == cls)[0]
                false_obj_idx = false_obj_idx[~_tp]
                for _ in false_obj_idx:
                    gt_class[_] = -1

            tmp_gt = [dict(_id=_id, image=image, bboxes=gt_box, classes=gt_class, confs=gt_conf)]
            tmp_det = [dict(image=image, bboxes=det_box, classes=det_class, confs=det_conf)]
            visualizer = DataVisualizer(save_dir, verbose=self.verbose, stdout_method=self.stdout_method)
            visualizer(tmp_gt, tmp_det, cls_alias=cls_alias)

        return ret1, ret2


confusion_matrix = ConfusionMatrix()
pr = PR()
ap = AP()
easy_metric = EasyMetric()
