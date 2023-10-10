from torch import nn
import torch.nn.functional as F


class BaseSemSegModel(nn.Module):
    def forward(self, x, pix_images=None):
        if self.training and pix_images is not None:
            return dict(
                preds=x,
                loss=self.loss(x, pix_images)
            )
        else:
            return self.post_process(x)

    def post_process(self, preds):
        return preds.argmax(1)

    def loss(self, preds, pix_images):
        """
        Args:
            preds: [b, out_features + 1, h, w]
            pix_images: [b, h, w]

        """
        # value=255 is the padding or edge areas
        return F.cross_entropy(preds, pix_images, ignore_index=255)
