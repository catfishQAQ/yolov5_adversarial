"""Modules for creating adversarial object patch."""

import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_patch_gen.utils.median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = [-0.25, 0.25],
        y_off_loc: Tuple[float, float] = [-0.25, 0.25],
        dev: torch.device = torch.device("cuda:0"),
    ):
        super(PatchTransformer, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self, adv_patch, lab_batch, model_in_sz, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True
    ):
        # add gaussian noise to reduce contrast with a stohastic process
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            # 添加高斯乘法和加法的扰动，使颜色亮度有差异，提升泛化能力。并使用种植滤波减少噪点
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        m_h, m_w = model_in_sz
        # Determine size of padding
        pad = (m_w - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        if do_transforms:
            # 为每个 patch 应用不同的对比度、亮度、噪声增强，限制在合法的像素亮度范围内
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # 根据标签信息（lab_batch）生成一个与 patch 相同 shape 的“掩码”张量，用来确定是否应该给某个目标贴 patch。 lab_batch[..., 0] 取出每个目标的 class_id，shape 为 [B, max_labels]
        cls_ids = lab_batch[..., 0].unsqueeze(-1)  # equiv to torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, p_c)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = self.tensor(cls_mask.size()).fill_(1)

        # Pad patch and mask to image dimensions
        #定义一个 给 patch 做上下左右 padding 的函数，对 patch 和 mask 做 padding
        patch_pad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0) 
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        #计算总目标数量，生成旋转角度，会在后面构建仿射变换矩阵 theta 时用到
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        #计算 patch 当前的大小，把标签从归一化坐标转为像素坐标，计算每个目标应该贴的 patch 大小
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = self.tensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * m_w
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * m_w
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * m_w
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * m_w
        #计算每个目标应该贴的 patch 大小，取出目标框中心点位置 + 偏移量
        tsize = np.random.uniform(*self.t_size_frac)
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(tsize)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(tsize)) ** 2)
        )

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        #为每个目标随机生成一个位置偏移，得到新的贴 patch 中心位置 (target_x, target_y)
        if rand_loc:
            off_x = targetoff_x * (self.tensor(targetoff_x.size()).uniform_(*self.x_off_loc))
            target_x = target_x + off_x
            off_y = targetoff_y * (self.tensor(targetoff_y.size()).uniform_(*self.x_off_loc))
            target_y = target_y + off_y
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)
        #将 [B, max_labels, ...] 拉平成 [B×max_labels, ...]，准备做变换
        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])
        #计算每个 patch 要移动到图像哪个位置，基于目标中心点位置
        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        #构造仿射变换矩阵 theta，把前面算出来的角度、缩放、位置写进 theta
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        #根据每个 theta 创建一个采样网格 grid，告诉系统：“我要把 patch 贴到哪里”，使用 grid_sample 执行变换
        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)
        #恢复原始维度，限制像素值范围
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #返回最终的贴图结果
        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """
    #默认值是 1，表示默认使用完全贴图（不透明 patch）
    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha
    #将每个目标的 patch（按位置对齐）贴到原图上
    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch
