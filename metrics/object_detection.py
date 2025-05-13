import numpy as np
from typing import List, Iterable
from tqdm import tqdm
from utils import os_lib


class Area:
    @staticmethod
    def areas(boxes):
        """Area(boxes) = (x2 - x1) * (y2 - y1)

        Args:
            boxes(np.array): shape=(N, 4), 4 means xyxy.

        Returns:
            rectangle_areas(np.array): shape=(N, )
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    @staticmethod
    def intersection_areas(boxes1, boxes2):
        """Area(boxes1 & boxes2)

        Args:
            boxes1(np.array): shape=(N, 4), 4 means xyxy.
            boxes2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            intersection_box(np.array): shape=(N, M)
        """
        # boxes1[:, None, 2:]: (N, 2) -> (N, 1, 2)
        # minimum((N, 1, 2), (M, 2)) -> broadcast ->  minimum((N, M, 2), (N, M, 2)) -> (N, M, 2)
        return (np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.maximum(boxes1[:, None, :2], boxes2[:, :2])).clip(0).prod(2)

    @classmethod
    def union_areas(cls, boxes1, boxes2, inter=None):
        """Area(boxes1 | boxes2)

        Arguments:
            boxes1(np.array): shape=(N, 4), 4 means xyxy.
            boxes2(np.array): shape=(M, 4), 4 means xyxy.

        Returns:
            union_areas(np.array): shape=(N, M)
        """
        area1 = cls.areas(boxes1)
        area2 = cls.areas(boxes2)
        if inter is None:
            inter = cls.intersection_areas(boxes1, boxes2)

        xv, yv = np.meshgrid(area1, area2)
        return xv.T + yv.T - inter

    @staticmethod
    def outer_areas(boxes1, boxes2):
        """outer rectangle area
        Area(boxes1 | boxes2) - Area(boxes1 & boxes2)

        Args:
            boxes1(np.array): shape=(N, 4), 4 means xyxy.
            boxes2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            outer_areas(np.array): shape=(N, M)
        """
        return (np.maximum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.minimum(boxes1[:, None, :2], boxes2[:, :2])).clip(0).prod(2)


class Area1D(Area):
    @staticmethod
    def intersection_areas(boxes1, boxes2):
        """boxes1 and boxes2 must have the same shape

        Args:
            boxes1: shape=(N, 4), 4 means xyxy.
            boxes2: shape=(N, 4), 4 means xyxy.

        Returns:
            intersection_box(np.array): shape=(N, )
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.T

        return (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    @classmethod
    def union_areas(cls, boxes1, boxes2, inter=None):
        """boxes1 and boxes2 must have the same shape

        Args:
            boxes1: shape=(N, 4), 4 means xyxy.
            boxes2: shape=(N, 4), 4 means xyxy.

        Returns:
            intersection_box(np.array): shape=(N, )
        """
        area1 = cls.areas(boxes1)
        area2 = cls.areas(boxes2)
        if inter is None:
            inter = cls.intersection_areas(boxes1, boxes2)

        return area1 + area2 - inter

    @staticmethod
    def outer_areas(boxes1, boxes2):
        """outer rectangle area

        Args:
            boxes1(np.array): shape=(N, 4), 4 means xyxy.
            boxes2(np.array): shape=(N, 4), 4 means xyxy.

        Returns:
            outer_areas(np.array): shape=(N, )
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.T
        return (np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)).clip(0) * (np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)).clip(0)


class PolygonArea:
    @staticmethod
    def areas(segmentations):
        """Shoelace Formula
        Area(boxes) = 0.5 * sum(x_i * y_(i+1) - y_i * x_(i+1))

        Args:
            segmentations(np.array): shape=(N, K, 2), 2 means xy.

        Returns:
            rectangle_areas(np.array): shape=(N, )
        """
        k, m, _ = segmentations.shape

        x = segmentations[:, :, 0]
        y = segmentations[:, :, 1]

        x_shifted = np.roll(x, -1, axis=1)
        y_shifted = np.roll(y, -1, axis=1)

        term1 = x * y_shifted
        term2 = y * x_shifted

        area = 0.5 * np.abs(np.sum(term1 - term2, axis=1))
        return area

    @staticmethod
    def intersection_areas(segmentations1, segmentations2):
        """Area(segmentations1 & segmentations2)

        Args:
            segmentations1: shape=(N, K1, 2), 2 means xy.
            segmentations2: shape=(M, K2, 2), 2 means xy.

        Returns:
            intersection_areas(np.array): shape=(N, M)
        """
        from shapely.geometry import Polygon

        areas = np.zeros((segmentations1.shape[0], segmentations2.shape[0]))
        for i, points1 in enumerate(segmentations1):
            for j, points2 in enumerate(segmentations2):
                polygon1 = Polygon(points1)
                polygon2 = Polygon(points2)

                intersection = polygon1.intersection(polygon2)
                areas[i, j] = intersection.area

        return areas

    @staticmethod
    def union_areas(segmentations1, segmentations2, *args, **kwargs):
        """Area(segmentations1 | segmentations2)

        Args:
            segmentations1: shape=(N, K1, 2), 2 means xy.
            segmentations2: shape=(M, K2, 2), 2 means xy.

        Returns:
            intersection_areas(np.array): shape=(N, M)
        """
        from shapely.geometry import Polygon

        areas = np.zeros((segmentations1.shape[0], segmentations2.shape[0]))
        for i, points1 in enumerate(segmentations1):
            for j, points2 in enumerate(segmentations2):
                polygon1 = Polygon(points1)
                polygon2 = Polygon(points2)

                intersection = polygon1.union(polygon2)
                areas[i, j] = intersection.area

        return areas


class Overlap:
    """only judge whether 2 object is overlap or not,
    for ConfusionMatrix of classification"""

    @staticmethod
    def point_in_line(points, lines):
        """point = a, line = (b1, b2)
        point do not fall in the line means that
            a after b1 and a before b2 (b1 < a < b2)

        Args:
            points: (n, )
            lines: (m, 2)
        """
        f = (points[:, None] > lines[None, :, 0]) & (points[:, None] < lines[None, :, 1])
        return f

    @staticmethod
    def point_in_line2D(points, lines):
        """point = (xa, ya), line = (xb1, yb1, xb2, yb2)
        point do not fall in the line means that

        Args:
            points: (n, 2)
            lines: (m, 4)
        """

    @staticmethod
    def line(lines1, lines2):
        """line1 = (a1, a2), line2 = (b1, b2),
        2 lines do not overlap means that
           a1 after b2 (a1 > b2)
        or a2 before b1 (a2 < b1)

        Args:
            lines1: (n, 2)
            lines2: (m, 2)
        """
        a1, a2 = lines1.T
        b1, b2 = lines2.T

        f = (a1[:, None] > b2[None, :]) | (a2[:, None] < b1[None, :])

        return ~f

    @staticmethod
    def line2D(lines1, lines2, return_insert_point=False):
        """line1 = (xa1, ya1, xa2, ya2), line2 = (xb1, yb1, xb2, yb2)

        Args:
            lines1: (n, 4)
            lines2: (m, 4)
            return_insert_point:
        """
        ab = np.repeat((lines1[:, (2, 3)] - lines1[:, (0, 1)])[None, :], len(lines1), axis=0)
        cd = np.repeat((lines2[:, (2, 3)] - lines2[:, (0, 1)])[:, None], len(lines2), axis=1)
        ac = lines2[:, None, (0, 1)] - lines1[None, :, (0, 1)]

        v1 = np.cross(cd, ac)
        v2 = np.cross(ab, ac)
        v3 = np.cross(cd, ab)

        t = v1 / v3
        u = v2 / v3

        flag = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if return_insert_point:
            p = lines1[:, (0, 1)][None, :] + ab * t[:, :, None]
            return flag, p

        else:
            return flag

    @staticmethod
    def line_in_box(lines, boxes):
        """line = (xa1, ya1, xa2, ya2), box = (xb1, yb1, xb2, yb2)

        Args:
            lines: (n, 4)
            boxes: (m, 4)
        """

    @staticmethod
    def box(boxes1, boxes2):
        """boxes1 = (xa1, ya1, xa2, ya2), boxes2 = (xb1, yb1, xb2, yb2)
        2 boxes do not overlap means that
           point 'a1' at the right or down of boxes2 (xa1 > xb2 | ya1 > yb2)
        or point 'a2' at the left or top of boxes2 (xa2 < xb1 | ya2 < yb1)

        Args:
            boxes1: (n, 4)
            boxes2: (m, 4)
        """

        xa1, ya1, xa2, ya2 = boxes1.T
        xb1, yb1, xb2, yb2 = boxes2.T

        f1 = (xa1[:, None] > xb2[None, :]) | (ya1[:, None] > yb2[None, :])
        f2 = (xa2[:, None] < xb1[None, :]) | (ya2[:, None] < yb1[None, :])

        return ~(f1 | f2)


class Iou:
    """
    Arguments:
        boxes1(np.array): shape=(N, 4), 4 means xyxy.
        boxes2(np.array): shape=(M ,4), 4 means xyxy.

    Returns:
        iou_mat(np.array): shape=(N, M)
    """

    def __init__(self, area_method=Area, alpha=1, eps=1e-6):
        """

        Args:
            alpha:
                alpha iou, https://arxiv.org/abs/2110.13675
                if used, set in 3 which is the default value of paper
            eps:
                note that, do not set too small, 'cause it will
                when run in low computational accuracy mode
                e.g. np.array(1e-12, dtype=np.float16) -> array(0., dtype=float16)
        """
        self.area_method = area_method
        self.alpha = alpha
        self.eps = eps

    def iou(self, boxes1, boxes2, inter=None, union=None):
        """vanilla iou
        Area(boxes1 & boxes2) / Area(boxes1 | boxes2)
        See Also `torchvision.ops.box_iou`


        """
        boxes1, boxes2 = np.array(boxes1), np.array(boxes2)

        if inter is None:
            inter = self.area_method.intersection_areas(boxes1, boxes2)

        if union is None:
            union = self.area_method.union_areas(boxes1, boxes2, inter)

        return inter / (union + self.eps)

    def u_iou(self, boxes1, boxes2, inter=None):
        """unidirectional iou
        Area(boxes1 & boxes2) / Area(boxes2)

        >>> a = np.array([[1, 1, 4, 4]])
        >>> b = np.array([[2, 2, 5, 5]])
        >>> Iou.u_iou(a, b) # ((4 - 2) * (4 - 2)) / (5 - 2) * (5 - 2)
        [[0.44444444]]
        """
        area2 = self.area_method.areas(boxes2)  # (M,)
        if inter is None:
            inter = self.area_method.intersection_areas(boxes1, boxes2)

        return inter / (area2 + self.eps)

    def b_iou(self, boxes1, boxes2, inter=None):
        """Bidirectional iou
        Area(boxes1 & boxes2) / min{Area(boxes1), Area(boxes2)}"""
        boxes1, boxes2 = np.array(boxes1), np.array(boxes2)

        if inter is None:
            inter = self.area_method.intersection_areas(boxes1, boxes2)

        u_iou1 = self.u_iou(boxes1, boxes2, inter=inter)
        u_iou2 = self.u_iou(boxes2, boxes1, inter=inter.T).T
        return np.maximum(u_iou1, u_iou2)

    def m_iou(self, boxes1, boxes2, inter=None):
        """mean iou
        (u_iou(boxes1, boxes2) + u_iou(boxes2, boxes1).T) / 2
        """
        boxes1, boxes2 = np.array(boxes1), np.array(boxes2)

        if inter is None:
            inter = self.area_method.intersection_areas(boxes1, boxes2)

        u_iou1 = self.u_iou(boxes1, boxes2, inter=inter)
        u_iou2 = self.u_iou(boxes2, boxes1, inter=inter.T).T

        return (u_iou1 + u_iou2) / 2

    def g_iou(self, boxes1, boxes2):
        """generalized iou, https://arxiv.org/pdf/1902.09630.pdf
        iou - (c - Area(boxes1 & boxes2)) / c
        See Also `torchvision.ops.generalized_box_iou`
        """
        outer = self.area_method.outer_areas(boxes1, boxes2)
        inter = self.area_method.intersection_areas(boxes1, boxes2)
        union = self.area_method.union_areas(boxes1, boxes2)
        iou = self.iou(boxes1, boxes2, inter=inter, union=union)

        return iou - (outer - union) / outer

    def d_iou(self, boxes1, boxes2, iou=None):
        """distance iou, https://arxiv.org/pdf/1911.08287.pdf
        iou - d ^ 2 / c ^ 2
        See Also `torchvision.ops.distance_box_iou`
        """
        boxes1_center = (boxes1[:, 2:] - boxes1[:, :2]) / 2
        boxes2_center = (boxes2[:, 2:] - boxes2[:, :2]) / 2

        d = np.linalg.norm(boxes1_center[:, None, :] - boxes2_center, axis=2) ** 2
        c = np.linalg.norm((np.maximum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.minimum(boxes1[:, None, :2], boxes2[:, :2])).clip(0), axis=2) ** 2 + self.eps

        if iou is None:
            iou = self.iou(boxes1, boxes2)

        return iou - d / c

    def c_iou(self, boxes1, boxes2, a=None, v=None):
        """complete iou, https://arxiv.org/pdf/1911.08287.pdf
        diou - av
        See Also `torchvision.ops.complete_box_iou`
        """
        iou = self.iou(boxes1, boxes2)
        diou = self.d_iou(boxes1, boxes2, iou=iou)

        if v is None:
            boxes1_wh = boxes1[:, 2:] - boxes1[:, :2]
            boxes2_wh = boxes2[:, 2:] - boxes2[:, :2]
            b1 = np.arctan(boxes1_wh[:, 0] / boxes1_wh[:, 1])
            b2 = np.arctan(boxes2_wh[:, 0] / boxes2_wh[:, 1])

            v = 4 / np.pi ** 2 * ((b1[:, None] - b2) ** 2)

        if a is None:
            a = v / (1 - iou + v + self.eps)

        return diou - a * v


class Iou1D(Iou):
    """
    Arguments:
        boxes1(np.array): shape=(N, 4), 4 means xyxy.
        boxes2(np.array): shape=(N ,4), 4 means xyxy.

    Returns:
        iou_mat(np.array): shape=(N, N)
    """

    def __init__(self, alpha=1, eps=1e-6):
        super().__init__(Area1D, alpha, eps)

    def d_iou(self, boxes1, boxes2, iou=None):
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.T

        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # outer width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # outer height
        c = cw ** 2 + ch ** 2 + self.eps
        d = ((b2_x1 - b1_x1 + b2_x2 - b1_x2) ** 2 + (b2_y1 - b1_y1 + b2_y2 - b1_y2) ** 2) / 4

        if iou is None:
            iou = self.iou(boxes1, boxes2)

        return iou - d / c

    def c_iou(self, boxes1, boxes2, a=None, v=None):
        iou = self.iou(boxes1, boxes2)
        d_iou = self.d_iou(boxes1, boxes2, iou=iou)

        if v is None:
            b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.T
            b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.T
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

            v = (4 / np.pi ** 2) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2)

        if a is None:
            a = v / (1 - iou + v + self.eps)

        return d_iou - a * v


class PolygonIou(Iou):
    """
    Arguments:
        segmentations1: shape=(N, K1, 2), 2 means xy.
        segmentations2: shape=(M, K2, 2), 2 means xy.

    Returns:
        iou_mat(np.array): shape=(N, N)
    """

    def __init__(self, alpha=1, eps=1e-6):
        super().__init__(PolygonArea, alpha, eps)


class ConfusionMatrix:
    def __init__(self, iou_method=None, **iou_method_kwarg):
        self.iou_method = iou_method or Iou(**iou_method_kwarg).iou

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
                tmp = np.concatenate([gt_box, det_box], 0)
                if tmp.size:
                    offset = np.max(tmp)
                    gt_box = gt_box + (true_class * offset)[:, None]
                    det_box = det_box + (pred_class * offset)[:, None]

            iou = self.iou_method(gt_box, det_box)

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
        gt_obj_idx = np.concatenate(gt_obj_idx).astype(np.int64)
        det_obj_idx = np.concatenate(det_obj_idx).astype(np.int64)

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
        return self._ap_thres(_results, confs)

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
            results = self._ap_thres(_results, confs)

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

    def _ap_thres(self, _results, confs=None):
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
    def __init__(self, iou_thres=0.5, cls_alias=None, verbose=False, stdout_method=print, **ap_kwargs):
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
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
                gt_iter_data = loader.load_full(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full(sub_dir=sub_dir, task=task)

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
                gt_iter_data = loader.load_full(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full(sub_dir=sub_dir, task=task)
                cls_alias = {}  # set if you want to convert class name

                EasyMetric(cls_alias=cls_alias).checkout_false_sample(gt_iter_data, det_iter_data, data_dir=data_dir, image_dir=image_dir, save_res_dir=save_res_dir)

        """
        from utils import os_lib
        from data_parse.cv_data_parse.datasets.base import DataVisualizer

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
            for i in tqdm(idx, desc=f'checkout {cls_alias[cls]}'):
                target_idx = det_obj_idx == i
                _tp = tp[target_idx]

                gt_class = gt_classes[i]
                gt_box = gt_boxes[i]
                conf = confs[i]
                det_class = det_classes[i]
                det_box = det_boxes[i]
                _id = _ids[i]

                image = os_lib.Loader(verbose=self.verbose).load_img(f'{rets[_id]["image_dir"]}/{_id}')

                false_obj_idx = np.where(det_class == cls)[0]
                false_obj_idx = false_obj_idx[~_tp]

                gt_class = [int(c) for c in gt_class]
                det_class = [int(c) for c in det_class]

                cls_alias[-1] = cls_alias[cls]

                for _ in false_obj_idx:
                    det_class[_] = -1

                tmp_ori = [dict(_id=_id, image=image)]
                tmp_gt = [dict(image=image, bboxes=gt_box, classes=gt_class)]
                tmp_det = [dict(image=image, bboxes=det_box, classes=det_class, confs=conf)]
                visualizer = DataVisualizer(save_dir, pbar=False, verbose=self.verbose, stdout_method=self.stdout_method)
                visualizer(tmp_ori, tmp_gt, tmp_det, cls_alias=cls_alias)

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
                gt_iter_data = loader.load_full(sub_dir='true_abs_xyxy')
                det_iter_data = loader.load_full(sub_dir=sub_dir, task=task)
                cls_alias = {}  # set if you want to convert class name

                EasyMetric(cls_alias=cls_alias).checkout_diff_sample(gt_iter_data, det_iter_data, data_dir=data_dir, image_dir=image_dir, save_res_dir=save_res_dir)

        """

        from utils import os_lib
        from data_parse.cv_data_parse.datasets.base import DataVisualizer

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

            image = os_lib.Loader(verbose=self.verbose).load_img(f'{rets[_id]["image_dir"]}/{_id}')

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

            tmp_ori = [dict(_id=_id, image=image)]
            tmp_gt = [dict(image=image, bboxes=gt_box, classes=gt_class, confs=gt_conf)]
            tmp_det = [dict(image=image, bboxes=det_box, classes=det_class, confs=det_conf)]
            visualizer = DataVisualizer(save_dir, pbar=False, verbose=self.verbose, stdout_method=self.stdout_method)
            visualizer(tmp_ori, tmp_gt, tmp_det, cls_alias=cls_alias)

        return ret1, ret2


confusion_matrix = ConfusionMatrix()
pr = PR()
ap = AP()
easy_metric = EasyMetric()
