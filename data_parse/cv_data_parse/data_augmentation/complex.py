import numpy as np
import cv2
from metrics.object_detection import Iou


class CopyPaste:
    """https://arxiv.org/abs/2012.07177"""

    def __init__(self, alpha=32., beta=32.):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image_list, bboxes_list, classes_list=None, **kwargs):
        img1, img2 = image_list
        bboxes1, bboxes2 = bboxes_list
        h, w = img1.shape[:2]

        # todo: wait for completing


class MixUp:
    """https://arxiv.org/pdf/1710.09412.pdf

    Args:
        alpha (float): mixup ratio
        beta (float): mixup ratio
    """

    def __init__(self, alpha=32., beta=32.):
        self.alpha = alpha
        self.beta = beta

    def get_add_params(self, r):
        return {'complex.MixUp': dict(r=r)}

    def parse_add_params(self, ret):
        return ret['complex.MixUp']['r']

    def get_params(self):
        return np.random.beta(self.alpha, self.beta)  # mixup ratio, default alpha=beta=32.0

    def __call__(self, image_list, bboxes_list=None, classes_list=None, **kwargs):
        """input image must have same shape"""
        info = self.get_add_params(*self.get_params())
        return dict(
            image=self.apply_image_list(image_list, info),
            bboxes=self.apply_bboxes_list(bboxes_list),
            classes=self.apply_classes_list(classes_list),
            **info
        )

    def apply_image_list(self, image_list, ret):
        img1, img2 = image_list
        r = self.parse_add_params(ret)
        image = (img1 * r + img2 * (1 - r)).astype(img1.dtype)
        return image

    def apply_bboxes_list(self, bboxes_list, *args):
        if bboxes_list is not None:
            bboxes = np.concatenate(bboxes_list, 0)
            return bboxes

    def apply_classes_list(self, classes_list, *args):
        if classes_list is not None:
            classes = np.concatenate(classes_list, 0)
            return classes


class CutMix:
    def __init__(self, alpha=32., beta=32.):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image_list, bboxes_list, classes_list=None, **kwargs):
        img1, img2 = image_list
        bboxes1, bboxes2 = bboxes_list
        h, w = img1.shape[:2]

        idx = np.random.choice(range(len(bboxes2)))
        target_bbox = bboxes2[idx]

        shift = target_bbox[:2] - target_bbox[-2:]

        x = np.random.randint(shift[0], w - shift[0])
        y = np.random.randint(shift[1], h - shift[1])
        r = np.random.beta(self.alpha, self.beta)  # mixup ratio, default alpha=beta=32.0

        img1[y:y + shift[1], x:x + shift[0]] = (
                img1[y:y + shift[1], x:x + shift[0]] * r
                + img2[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] * (1 - r)
        ).astype(img1.dtype)

        bboxes = np.append(bboxes1, target_bbox)

        return dict(
            image=img1,
            bboxes=bboxes,
        )


class Mosaic4:
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.border = [img_size // 2, img_size // 2]

    def get_add_params(self, coors):
        return {'complex.Mosaic4': dict(coors=coors)}

    def parse_add_params(self, ret):
        return ret['complex.Mosaic4']['coors']

    def get_params(self, image_list):
        s = self.img_size
        yc, xc = (int(np.random.uniform(x, 2 * s - x)) for x in self.border)  # center x, y of img4
        coors = []
        for i, img in enumerate(image_list):
            h, w = img.shape[:2]

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            coors.append(((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b)))

        return coors

    def __call__(self, image_list, bboxes_list=None, classes_list=None, **kwargs):
        coors = self.get_params(image_list)
        info = self.get_add_params(coors)

        return dict(
            image=self.apply_image_list(image_list, info),
            bboxes=self.apply_bboxes_list(bboxes_list, info),
            classes=self.apply_classes_list(classes_list),
            **info
        )

    def apply_image_list(self, image_list, ret):
        s = self.img_size
        img4 = np.full((s * 2, s * 2, image_list[0].shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
        coors = self.parse_add_params(ret)

        for img, ((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b)) in zip(image_list, coors):
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        return img4

    def apply_bboxes_list(self, bboxes_list, ret):
        bboxes4 = None
        if bboxes_list is not None:
            coors = self.parse_add_params(ret)
            bboxes4 = []
            for bboxes, ((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b)) in zip(bboxes_list, coors):
                shift = np.array([x1a - x1b, y1a - y1b, x1a - x1b, y1a - y1b])
                bboxes += shift
                bboxes4.append(bboxes)
            bboxes4 = np.concatenate(bboxes4)
        return bboxes4

    def apply_classes_list(self, classes_list, *args):
        classes = None
        if classes_list is not None:
            classes = np.concatenate(classes_list)

        return classes


class Mosaic9:
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.border = [img_size // 2, img_size // 2]

    def get_add_params(self, coors, yc, xc):
        return {'complex.Mosaic4': dict(coors=coors, yc=yc, xc=xc)}

    def parse_add_params(self, ret):
        info = ret['complex.Mosaic4']
        return info['coors'], info['yc'], info['xc']

    def get_params(self, image_list):
        s = self.img_size
        hp, wp = -1, -1  # height, width previous
        coors = []
        for i, img in enumerate(image_list):
            h, w = img.shape[:2]

            # place img in img9
            if i == 0:  # center
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            else:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords
            hp, wp = h, w  # height, width previous

            coors.append(((x1, y1, x2, y2), (padx, pady)))
        yc, xc = (int(np.random.uniform(0, s)) for _ in self.border)  # mosaic center x, y

        return coors, yc, xc

    def __call__(self, image_list, bboxes_list=None, classes_list=None, **kwargs):
        info = self.get_add_params(*self.get_params(image_list))

        return dict(
            image=self.apply_image_list(image_list, info),
            bboxes=self.apply_bboxes_list(bboxes_list, info),
            classes=self.apply_classes_list(classes_list),
            **info
        )

    def apply_image_list(self, image_list, ret):
        coors, yc, xc = self.parse_add_params(ret)
        s = self.img_size
        img9 = np.full((s * 3, s * 3, image_list[0].shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

        for img, ((x1, y1, x2, y2), (padx, pady)) in zip(image_list, coors):
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]

        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
        return img9

    def apply_bboxes_list(self, bboxes_list, ret):
        bboxes9 = None
        if bboxes_list is not None:
            bboxes9 = []
            coors, yc, xc = self.parse_add_params(ret)
            for bboxes, ((x1, y1, x2, y2), (padx, pady)) in zip(bboxes_list, coors):
                shift = np.array([padx, pady, padx, pady])
                bboxes += shift
                bboxes9.append(bboxes)

            bboxes9 = np.concatenate(bboxes9)
            bboxes9[:, [0, 2]] -= xc
            bboxes9[:, [1, 3]] -= yc

        return bboxes9

    def apply_classes_list(self, classes_list, *args):
        classes = None
        if classes_list is not None:
            classes = np.concatenate(classes_list)

        return classes
