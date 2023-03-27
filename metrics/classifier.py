import numpy as np


class ConfusionMatrix:
    @staticmethod
    def tp(true, pred, pos=1):
        """true positive"""
        return np.sum((true == pos) & (pred == pos))

    @staticmethod
    def fp(true, pred, pos=1):
        """false positive"""
        return np.sum((true != pos) & (pred == pos))

    @staticmethod
    def fn(true, pred, pos=1):
        """false negative"""
        return np.sum((true == pos) & (pred != pos))

    @staticmethod
    def tn(true, pred, pos=1):
        """true positive"""
        return np.sum((true != pos) & (pred != pos))


class PR:
    @staticmethod
    def tpr(true, pred, pos=1):
        """tpr"""
        tp = ConfusionMatrix.tp(true, pred, pos)

        return tp / np.sum(true == pos)

    recall = tpr

    @staticmethod
    def ppv(true, pred, pos=1):
        """ppv"""
        tp = ConfusionMatrix.tp(true, pred, pos)

        return tp / np.sum(pred == pos)

    precision = ppv

    @staticmethod
    def fpr(true, pred, pos=1):
        fp = ConfusionMatrix.fp(true, pred, pos)
        return fp / np.sum(true != pos)

    @staticmethod
    def acc(true, pred, pos=1):
        tp = ConfusionMatrix.tp(true, pred, pos)
        tn = ConfusionMatrix.fn(true, pred, pos)

        return (tp + tn) / len(true)


class TopTarget:
    @staticmethod
    def f_measure(true, pred, pos=1, a=1):
        p = PR.precision(true, pred, pos)
        r = PR.recall(true, pred, pos)

        return dict(
            p=p,
            r=r,
            f=(a ** 2 + 1) * p * r / (a ** 2 * p + r)
        )

    @classmethod
    def f1(cls, true, pred, pos=1):
        return cls.f_measure(true, pred, pos)

    @staticmethod
    def pr_curve(true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = []
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            p = PR.precision(true_, pred_)
            r = PR.precision(true_, pred_)

            pr.append((p, r, thres))

        p = np.array([_[0] for _ in pr])
        r = np.array([_[1] for _ in pr])
        best_idx = np.argmin(np.abs(p / r - 1))

        return dict(
            pr=pr,
            best_idx=best_idx
        )

    @staticmethod
    def roc(true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = []
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            tpr = PR.tpr(true_, pred_)
            fpr = PR.fpr(true_, pred_)

            pr.append((tpr, fpr, thres))

        pr_ = np.array(pr)[:, :2]
        base = np.ones_like(pr_)

        cos_sim = pr_ * base / np.linalg.norm(pr_, axis=1) * np.sqrt(2)
        best_idx = np.argmin(cos_sim)

        return dict(
            pr=pr,
            best_idx=best_idx
        )

    @staticmethod
    def auc(true, pred):
        pass

    @staticmethod
    def ks(true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = []
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            tpr = PR.tpr(true_, pred_)
            fpr = PR.fpr(true_, pred_)

            pr.append((tpr, fpr, thres))

        pr_ = np.array(pr)[:, :2]
        pr_[1] = 1 - pr_[1]
        base = np.ones_like(pr_)

        cos_sim = pr_ * base / np.linalg.norm(pr_, axis=1) * np.sqrt(2)
        best_idx = np.argmin(cos_sim)

        return dict(
            pr=pr,
            best_idx=best_idx
        )
