import torch
from torch import nn
from utils import torch_utils
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def forward(self, student_loss, student_logits, teacher_logits, alpha=0.):
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )
        loss = student_loss * alpha + kl_loss * (1 - alpha)

        return {
            'loss': loss,
            'loss.student': student_loss,
            'loss.kl': kl_loss,
        }


class ModelWrap:
    """
    Usages:
        .. code-block:: python

            model = Model()
            teacher = TeacherModel()
            model_wrap = ModelWrap(teacher)
            model_wrap.wrap(model)

            # define your train step
            ...
            out = model(data)
            ...
    """

    def __init__(self, teacher, alpha=0.):
        self.teacher = teacher
        self.model = None
        self.alpha = alpha
        self.criterion = DistillLoss()
        torch_utils.ModuleManager.freeze_module(self.teacher, allow_train=False)

    def wrap(self, model: nn.Module):
        ori_fit = model.fit

        def fit(*args, **kwargs):
            with torch.no_grad():
                teacher_out = self.teacher(*args, **kwargs)
                teacher_logits = teacher_out['logits']
            student_out = ori_fit(*args, **kwargs)
            student_loss = student_out['loss']
            student_logits = student_out['logits']
            return self.criterion(student_loss, student_logits, teacher_logits, self.alpha)

        model.fit = fit
        return model

    def dewrap(self):
        raise NotImplementedError
