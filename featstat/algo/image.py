from functools import lru_cache

import math

import numpy as np
import quaternion  # adds to numpy  # noqa # pylint: disable=unused-import
import cv2
from scipy.optimize import leastsq


class ImageProc:
    latest_opt = None
    show_fit = None

    @staticmethod
    def add_noise_to_image(image, noise_img_file):
        tmp = cv2.imread(noise_img_file, cv2.IMREAD_UNCHANGED)
        noise_img = cv2.resize(tmp, None,
                               fx=image.shape[1] / tmp.shape[1],
                               fy=image.shape[0] / tmp.shape[0],
                               interpolation=cv2.INTER_CUBIC)

        return cv2.add(image, noise_img[:, :, 3])

    @staticmethod
    def crop_and_zoom_image(image, x_off, y_off, width, height, scale, trg_w_h=None, depth=None):
        tw, th = trg_w_h
        if scale is None:
            scale = min(th / height, tw / width)

        res = []
        for img in [image] + ([] if depth is None else [depth]):
            imgc = cv2.resize(img[y_off:y_off + height, x_off:x_off + width], None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
            oh, ow = img.shape
            ch, cw = imgc.shape

            if trg_w_h is not None:
                if x_off + width >= ow:
                    x0 = tw - cw
                elif x_off <= 0:
                    x0 = 0
                else:
                    x0 = (tw - cw) // 2

                if y_off + height >= oh:
                    y0 = th - ch
                elif y_off <= 0:
                    y0 = 0
                else:
                    y0 = (th - ch) // 2

                imgd = np.zeros((th, tw), dtype=img.dtype)
                imgd[y0:y0 + ch, x0:x0 + cw] = imgc
            else:
                imgd = imgc
            res.append(imgd)

        if depth is not None:
            return res
        return res[0]

    @staticmethod
    def single_object_bounds(img, threshold, crop_marg, min_px, debug=False):
        # binary image
        _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # remove stars
        mask = cv2.erode(mask, ImageProc.bsphkern(9), iterations=1)

        if np.sum(mask) < min_px:
            return (None,) * 4

        # detect target
        x_, y_, w_, h_ = cv2.boundingRect(mask)

        # add margin
        x, y = max(0, x_ - crop_marg), max(0, y_ - crop_marg)
        w = min(mask.shape[1] - x, w_ + 2 * crop_marg - (x - x_ + crop_marg))
        h = min(mask.shape[0] - y, h_ + 2 * crop_marg - (y - y_ + crop_marg))

        if debug:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_color = cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            img_color[y + h // 2, x + w // 2] = (0, 0, 255)
            cv2.imshow('box', cv2.resize(img_color, (512, 512)))

        return x, y, w, h

    @staticmethod
    def equalize_brightness(image, ref_image, percentile=98, image_gamma=1):
        image = ImageProc.adjust_gamma(image, 1 / image_gamma)
        ip = np.percentile(image, percentile)
        rp = np.percentile(ImageProc.adjust_gamma(ref_image, 1 / image_gamma), percentile)
        image = cv2.convertScaleAbs(image, None, rp / ip, 0)
        return ImageProc.adjust_gamma(image, image_gamma)

    @staticmethod
    def default_preprocess(image, max=255):
        bg = np.percentile(image, 250 / 1024 * 100)
        return ImageProc.adjust_gamma(np.clip((image - bg) * max / (max - bg), 0, max), 1.8)

    @staticmethod
    def change_color_depth(img, src_bits, dst_bits):
        if src_bits == dst_bits:
            return img

        if str(img.dtype)[:4] == 'uint':
            new_type = 'uint' + str(math.ceil(dst_bits / 8) * 8)
        else:
            new_type = img.dtype

        if src_bits < dst_bits:
            img = img.astype(new_type)

        img = img * (2 ** (dst_bits - src_bits))

        if src_bits > dst_bits:
            img = img.astype(new_type)

        return img

    @staticmethod
    def remove_bg(img, bg_img, gain=None, max_val=None, offset=0):
        if gain is None:
            # estimate correct gain
            cost_fun = lambda g: np.var((img - g[0] * bg_img).reshape((-1, 3)), axis=0)
            res = leastsq(cost_fun, np.array([1]))
            gain = res[0][0]
            # TODO: check that gain is correct, then remove below line
            assert False, 'check that is %s correct gain from leastsq res: %s' % (gain, res)
            print('estimated bg gain: %f' % gain)

        imgr = img.astype('float') - gain * bg_img
        if offset not in (None, False):
            imgr += offset - np.min(imgr)
        if max_val and offset is not False:
            return np.clip(imgr, 0, max_val).astype(img.dtype)
        return imgr

    @staticmethod
    def color_correct(img, bgr_mx, inverse=False, max_val=None):
        assert img.shape[2] == 3, 'can only do to BGR images'
        if inverse:
            bgr_mx = np.linalg.inv(bgr_mx)

        imgc = bgr_mx.dot(img.reshape((-1, 3)).T).T.reshape(img.shape)
        if max_val:
            return np.clip(imgc, 0, max_val).astype(img.dtype)
        return imgc

    @staticmethod
    def adjust_gamma(image, gamma, gamma_break=None, linear_part=True, inverse=False, max_val=255):
        if gamma == 1:
            return image

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = gamma if inverse else 1.0 / gamma
        gamma_break = gamma_break or 0

        if image.dtype == 'uint8' and gamma_break == 0:
            # apply gamma correction using the lookup table
            max_val = min(max_val, 255)
            table = np.array([((i / max_val) ** invGamma) * max_val for i in np.arange(0, max_val + 1)]).astype(
                image.dtype)
            adj_img = cv2.LUT(image, table)
        elif gamma_break == 0:
            adj_img = np.round(((image / max_val) ** invGamma) * max_val).astype(image.dtype)
        elif True:
            # from https://se.mathworks.com/help/vision/ref/gammacorrection.html
            b_p = gamma_break
            s_ls = 1 / (gamma / b_p ** (1 / gamma - 1) - gamma * gamma_break + gamma_break)
            f_s = gamma * s_ls / b_p ** (1 / gamma - 1)
            c_o = f_s * b_p ** (1 / gamma) - s_ls * b_p

            img = image.flatten() / max_val
            I = img <= (s_ls if inverse else 1) * b_p
            nI = np.logical_not(I)

            adj_img = np.zeros(image.shape).flatten()
            adj_img[I] = (img[I] / s_ls) if inverse else (img[I] * s_ls)
            adj_img[nI] = (((img[nI] + c_o) / f_s) ** gamma) if inverse else (f_s * img[nI] ** (1 / gamma) - c_o)
            adj_img = (adj_img * max_val).reshape(image.shape).astype(image.dtype)

        else:
            # from https://en.wikipedia.org/wiki/SRGB
            if 1:
                a = gamma_break
                K0 = a / (gamma - 1)
            else:
                K0 = gamma_break
                a = K0 * (gamma - 1)

            alpha = 1 + a
            th = alpha ** gamma * (gamma - 1) ** (gamma - 1) / a ** (gamma - 1) / gamma ** gamma

            lim = K0 if inverse else K0 / th
            img = image.flatten() / max_val
            I = img <= lim
            nI = np.logical_not(I)

            adj_img = np.zeros(image.shape).flatten()
            adj_img[I] = (img[I] / th) if inverse else (th * img[I])
            adj_img[nI] = (((img[nI] + a) / alpha) ** gamma) if inverse else (alpha * img[nI] ** (1 / gamma) - a)
            adj_img = (adj_img * max_val).reshape(image.shape).astype(image.dtype)
            # adj_img = np.round(adj_img * max_val).reshape(image.shape).astype(image.dtype)

        return adj_img

    @staticmethod
    def apply_point_spread_fn(img, ratio):
        # ratio is how many % of power on central pixel
        sd = 1 / math.sqrt(2 * math.pi * ratio)
        size = 1 + 2 * math.ceil(sd * 2)
        kernel = ImageProc.gkern2d(size, sd)
        cv2.filter2D(img, -1, kernel, img)
        return img

    @staticmethod
    @lru_cache(maxsize=5)
    def gkern2d(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """
        w, h = (l[0], l[1]) if '__iter__' in dir(l) else (l, l)
        sx, sy = (sig[0], sig[1]) if '__iter__' in dir(sig) else (sig, sig)
        ax = np.arange(-w // 2 + 1., w // 2 + 1.)
        ay = np.arange(-h // 2 + 1., h // 2 + 1.)
        xx, yy = np.meshgrid(ax, ay)
        kernel = np.exp(-((xx / sx) ** 2 + (yy / sy) ** 2) / 2)
        return kernel / np.sum(kernel)

    @staticmethod
    def bsphkern(l=5):
        """
        creates a binary spherical kernel
        """
        gkern = ImageProc.gkern2d(l=l, sig=l)
        limit = gkern[l // 2 if isinstance(l, int) else l[1] // 2, -1] * 0.995
        return np.array(gkern >= limit, dtype=np.uint8)

    @staticmethod
    def fuzzy_kernel(kernel, sig):
        w = int(sig // 2)
        skernel = np.zeros(tuple(np.array(kernel.shape[:2]) + int(sig)) + kernel.shape[2:3], dtype=kernel.dtype)
        skernel[w:w + kernel.shape[0], w:w + kernel.shape[1]] = kernel
        gkrn = ImageProc.gkern2d(sig, sig / 2)
        skernel = cv2.filter2D(skernel, kernel.shape[2], gkrn)
        return skernel

    @staticmethod
    def _img_max_valid(img):
        max = 1.0 if 'float' in str(img.dtype) else 255
        assert max != 255 or img.dtype == np.uint8, 'wrong datatype for image: %s' % img.dtype
        return max

    @staticmethod
    def add_stars(img, mask, coef=2, cache=False):
        # add power law distributed stars to image
        assert img.shape == img.shape[:2], 'works only with grayscale images'
        if not cache:
            ImageProc._cached_random_stars.cache_clear()
        stars = ImageProc._cached_random_stars(coef, img.shape)
        # can be over 255, will clip later
        img[mask] = np.clip(stars[mask], 0, 600)
        return img

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_random_stars(coef, shape):
        return np.random.pareto(coef, shape)

    @staticmethod
    def add_sensor_noise(img, mean=7, sd=2, cache=False):
        if not cache:
            ImageProc._cached_sensor_noise.cache_clear()
        img += ImageProc._cached_sensor_noise(mean, sd, img.shape)
        return img

    @staticmethod
    @lru_cache(maxsize=1)
    def _cached_sensor_noise(mean, sd, shape):
        return np.random.normal(mean, sd, shape)

    @staticmethod
    def process_target_image(image_src, threshold_value=50):
        hist = cv2.calcHist([image_src], [0], None, [256], [0, 256])
        th, image_dst = cv2.threshold(image_src, threshold_value, 255, cv2.THRESH_TOZERO)
        return image_dst, hist, threshold_value

    @staticmethod
    def overlay_mask(image, mask):
        sc_img = min(image.shape[0], mask.shape[0])/image.shape[0]
        sc_mask = min(image.shape[0], mask.shape[0])/mask.shape[0]
        img_color = cv2.cvtColor(cv2.resize(image, None, fx=sc_img, fy=sc_img, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        mask_color = cv2.cvtColor(cv2.resize((mask > 0).astype(np.uint8)*255, None, fx=sc_mask, fy=sc_mask, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        mask_color[:, :, 0:2] = 0
        return cv2.addWeighted(img_color, 0.5, mask_color, 0.5, 0.0)

    @staticmethod
    def merge(images):
        summed_weights = 1
        summed_images = images[0]
        for i in range(1, len(images)):
            summed_images = cv2.addWeighted(summed_images, summed_weights / (summed_weights + 1),
                                            images[i], 1 / (summed_weights + 1), 0.0)
            summed_weights += 1
        return summed_images
