from torch import nn
import torch.nn.functional as F


class BaseSemSegModel(nn.Module):
    def forward(self, x, label_masks=None):
        if self.training and label_masks is not None:
            return dict(
                logits=x,
                loss=self.loss(x, label_masks)
            )
        else:
            return self.post_process(x)

    def post_process(self, logits):
        return logits.argmax(1)

    def loss(self, logits, label_masks):
        """
        Args:
            logits: [b, out_features + 1, h, w]
            label_masks: [b, h, w]

        """
        # value=255 is the padding or edge areas
        return F.cross_entropy(logits, label_masks, ignore_index=255)
