import numpy as np
from collections import defaultdict


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


class Iou:
    @staticmethod
    def line_iou(line1, line2):
        """line1 = (a1, b1), line2 = (a2, b2),
        which were overlap in 1-d cord where
        (a1 < a2 & b1 > a2) | (b1 > a2 & b1 < b2)
        """
        a1, b1, a2, b2 = line1[:, 0], line1[:, 1], line2[:, 0], line2[:, 1]

        f1 = (a1[:, None] <= a2[None, :]) & (b1[:, None] >= a2[None, :])
        f2 = (b1[:, None] >= a2[None, :]) & (b1[:, None] <= b2[None, :])

        return f1 | f2

    @staticmethod
    def vanilla(box1, box2, inter=None, union=None):
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
        iou = cls.vanilla(box1, box2, inter=inter, union=union)

        return iou - (outer - union) / outer

    @classmethod
    def diou(cls, box1, box2, iou=None):
        """https://arxiv.org/pdf/1911.08287.pdf
        iou - d ^ 2 / c ^ 2
        See Also `torchvision.ops.distance_box_iou`
        """
        box1_center = (box1[:, 2:] - box1[:, :2]) / 2
        box2_center = (box2[:, 2:] - box2[:, :2]) / 2

        d = np.linalg.norm(box1_center[:, None, :] - box2_center, axis=2) ** 2
        c = np.linalg.norm((np.maximum(box1[:, None, 2:], box2[:, 2:]) - np.minimum(box1[:, None, :2], box2[:, :2])).clip(0), axis=2) ** 2
        if iou is None:
            iou = cls.vanilla(box1, box2)

        return iou - d / c

    @classmethod
    def ciou(cls, box1, box2, a=None, v=None):
        """https://arxiv.org/pdf/1911.08287.pdf
        diou - av
        See Also `torchvision.ops.complete_box_iou`
        """
        iou = cls.vanilla(box1, box2)
        diou = cls.diou(box1, box2, iou=iou)

        if v is None:
            box1_wh = box1[:, 2:] - box1[:, :2]
            box2_wh = box2[:, 2:] - box2[:, :2]
            b1 = np.arctan(box1_wh[:, 0] / box1_wh[:, 1])
            b2 = np.arctan(box2_wh[:, 0] / box2_wh[:, 1])

            v = 4 / np.pi ** 2 * ((b1[:, None] - b2) ** 2)

        if a is None:
            a = v / (1 - iou + v)

        return diou - a * v


class ConfusionMatrix:
    @staticmethod
    def tp(gt_box, det_box, classes=None, iou_thres=0.5,
           iou=None, iou_method=Iou.vanilla, iou_method_kwarg=dict()):
        """

        Args:
            gt_box: (N, 4)
            det_box: (M, 4)
            classes (List[iter]):
            iou_thres:
            iou: (N, M)
            iou_method:
            iou_method_kwarg:

        Returns:
            tp: (M, )
            iou: (N, M)

        """
        if iou is None:
            iou = iou_method(gt_box, det_box, **iou_method_kwarg)
            iou = np.where(iou < iou_thres, 0, 1)

        if classes:
            true_class, pred_class = classes

            iou = iou_method(gt_box, det_box, **iou_method_kwarg)
            x = np.where((iou >= iou_thres))
            correct_index = np.where(true_class[x[0]] == pred_class[x[1]])

            x = np.stack(x, axis=1)
            x = x[correct_index[0], :]

            if x.shape[0]:
                matches = np.concatenate((x, iou[x[:, 0], x[:, 1]][:, None]), 1)  # [gt_box, det_box, iou]
                if x.shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            else:
                matches = np.zeros((0, 3))

            tp = np.zeros(det_box.shape[0], dtype=bool)
            tp[matches[:, 1].astype(int)] = True

            return tp, iou

        else:
            return np.sum(iou, axis=0, dtype=bool), iou

    @classmethod
    def fp(cls, *args, **kwargs):
        """see also ConfusionMatrix.tp"""
        tp, iou = cls.tp(*args, **kwargs)
        return -tp, iou


class PR:
    @staticmethod
    def recall(gt_box=None, det_box=None, conf=None, tp=None, n_true=None, eps=1e-16,
               iou_thres=0.5, iou_method=Iou.vanilla, iou_method_kwarg=dict()):
        """

        Args:
            gt_box:
            det_box:
            conf:
            tp: if set, `gt_box`, `det_box`, `conf` is not necessary, but n_true must be set
            n_true:
            iou_thres:
            iou_method:
            iou_method_kwarg:

        Returns:

        """
        sort_idx = None
        if tp is None:
            sort_idx = np.argsort(-conf)
            det_box = det_box[sort_idx]
            tp, iou = ConfusionMatrix.tp(gt_box, det_box, iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg)
            n_true = len(gt_box)

        acc_tp = np.cumsum(tp)

        return acc_tp / (n_true + eps), tp, sort_idx

    @staticmethod
    def precision(gt_box=None, det_box=None, conf=None, tp=None, eps=1e-16,
                  iou_thres=0.5, iou_method=Iou.vanilla, iou_method_kwarg=dict()):
        """

        Args:
            gt_box:
            det_box:
            conf:
            tp: if set, `gt_box`, `det_box`, `conf` is not necessary
            iou_thres:
            iou_method:
            iou_method_kwarg:

        Returns:

        """
        sort_idx = None
        if tp is None:
            sort_idx = np.argsort(-conf)
            det_box = det_box[sort_idx]
            tp, iou = ConfusionMatrix.tp(gt_box, det_box, iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg)

        acc_tp = np.cumsum(tp)
        n = np.cumsum(np.ones_like(acc_tp))

        return acc_tp / (n + eps), tp, sort_idx


class AP:
    """https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/paper_survey_on_performance_metrics_for_object_detection_algorithms.pdf"""

    @classmethod
    def ap_thres(
            cls, gt_box=None, det_box=None, conf=None, classes=None,
            tp=None, n_true=None,
            iou_thres=0.5, ap_method=None, ap_method_kwargs=dict(),
            iou_method=Iou.vanilla, iou_method_kwarg=dict(),
            return_more_info=True, **kwargs
    ):
        """AP@0.5"""
        ap_method = ap_method or cls.interp

        if classes is None:
            return cls.per_class_ap(
                gt_box, det_box, conf, tp, n_true=n_true,
                iou_thres=iou_thres, ap_method=ap_method, ap_method_kwargs=ap_method_kwargs,
                iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
            )
        else:
            unique_class = np.unique(np.concatenate(classes))
            true_class, pred_class = classes

            if tp is None:
                tp, iou = ConfusionMatrix.tp(gt_box, det_box, classes,
                                             iou_thres=iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg)

            sort_idx = np.argsort(-conf)
            tmp_tp = tp[sort_idx]
            tmp_pred_class = pred_class[sort_idx]

            results = {}

            for c in unique_class:
                r = cls.per_class_ap(
                    tp=tmp_tp[tmp_pred_class == c], n_true=len(gt_box[true_class == c]),
                    iou_thres=iou_thres, ap_method=ap_method, ap_method_kwargs=ap_method_kwargs,
                    iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
                )

                if return_more_info:
                    r.update(
                        sort_idx=sort_idx[pred_class == c],
                        det_box=det_box[pred_class == c],
                        gt_box=gt_box[true_class == c],
                        conf=conf[pred_class == c]
                    )

                results[c] = r

            return results

    @staticmethod
    def per_class_ap(
            gt_box=None, det_box=None, conf=None,
            tp=None, n_true=None,
            iou_thres=0.5, ap_method=None, ap_method_kwargs=dict(),
            iou_method=Iou.vanilla, iou_method_kwarg=dict()
    ):

        recall, tp, sort_idx = PR.recall(
            gt_box, det_box, conf, tp, n_true=n_true,
            iou_thres=iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
        )
        precision, tp, _ = PR.precision(
            gt_box, det_box, conf, tp,
            iou_thres=iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
        )

        # Append sentinel values to beginning and end
        mean_recall = np.concatenate(([0.0], recall, [1.0]))
        mean_precision = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mean_precision = np.flip(np.maximum.accumulate(np.flip(mean_precision)))

        ap = ap_method(mean_recall, mean_precision, **ap_method_kwargs)

        return dict(
            ap=ap,
            tp=tp,
            sort_idx=sort_idx,
            recall=recall,
            precision=precision,
            mean_recall=mean_recall,
            mean_precision=mean_precision
        )

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

    @classmethod
    def ap_thres_range(
            cls, gt_box, det_box, conf, classes=None,
            tp=None, n_true=None, thres_range=np.arange(0.5, 1, 0.05),
            iou_method=Iou.vanilla, iou_method_kwarg=dict(),
            ap_method=None, ap_method_kwargs=dict()
    ):
        """AP@0.5:0.95"""
        per_thres_result = {}

        for thres in thres_range:
            result = cls.ap_thres(
                gt_box, det_box, conf, classes=classes, tp=tp, n_true=n_true,
                iou_thres=thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg,
                ap_method=ap_method, **ap_method_kwargs)

            per_thres_result[thres] = result

        if classes is None:
            ap = np.mean([result['ap'] for result in per_thres_result.values()])
        else:
            ap = defaultdict(float)
            for thres, results in per_thres_result.items():
                for class_, result in results.items():
                    ap[class_] += result['ap']

            ap = {class_: s / len(per_thres_result) for class_, s in ap.items()}

        return dict(
            ap=ap,
            per_thres_result=per_thres_result,
        )

    @classmethod
    def mAP(
            cls, gt_boxes, det_boxes, confs, classes=None,
            iou_thres=0.5, ap_method=None, ap_method_kwargs=dict(),
            iou_method=Iou.vanilla, iou_method_kwarg=dict(),
            return_more_info=True,
    ):
        if classes is None:
            tps = []
            for data in zip(gt_boxes, det_boxes):
                tp, iou = ConfusionMatrix.tp(
                    *data, iou_thres=iou_thres,
                    iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
                )

                tps.append(tp)

            tps = np.concatenate(tps, axis=0)

        else:
            tmp_true_classes, tmp_pred_classes = [], []
            tps = []
            for data in zip(gt_boxes, det_boxes, classes):
                class_ = data[2]
                tp, iou = ConfusionMatrix.tp(
                    *data,
                    iou_thres=iou_thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg
                )
                tps.append(tp)
                tmp_true_classes.append(class_[0])
                tmp_pred_classes.append(class_[1])

            tps = np.concatenate(tps, axis=0)
            classes = (np.concatenate(tmp_true_classes), np.concatenate(tmp_pred_classes))

        det_boxes = np.concatenate(det_boxes, axis=0)
        gt_boxes = np.concatenate(gt_boxes, axis=0)
        confs = np.concatenate(confs, axis=0)

        results = cls.ap_thres(
            gt_boxes, det_boxes, confs, classes=classes, tp=tps, n_true=len(gt_boxes),
            iou_thres=iou_thres, ap_method=ap_method, ap_method_kwargs=ap_method_kwargs,
            iou_method=iou_method, iou_method_kwarg=iou_method_kwarg,
            return_more_info=return_more_info
        )

        return results

    @classmethod
    def mAP_thres_range(
            cls, gt_boxes, det_boxes, confs, classes=None,
            thres_range=np.arange(0.5, 1, 0.05), ap_method=None, ap_method_kwargs=dict(),
            iou_method=Iou.vanilla, iou_method_kwarg=dict()
    ):
        per_thres_result = {}

        for thres in thres_range:
            result = cls.mAP(
                gt_boxes, det_boxes, confs, classes,
                return_more_info=False,
                iou_thres=thres, iou_method=iou_method, iou_method_kwarg=iou_method_kwarg,
                ap_method=ap_method, **ap_method_kwargs
            )

            per_thres_result[thres] = result

        if classes is None:
            ap = np.mean([result['ap'] for result in per_thres_result.values()])
        else:
            ap = defaultdict(float)
            for thres, results in per_thres_result.items():
                for class_, result in results.items():
                    ap[class_] += result['ap']

            ap = {class_: s / len(per_thres_result) for class_, s in ap.items()}

        return dict(
            ap=ap,
            per_thres_result=per_thres_result,
        )
