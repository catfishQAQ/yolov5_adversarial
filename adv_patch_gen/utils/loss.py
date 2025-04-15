"""Loss functions used in patch generation."""

from typing import Tuple

import torch
import torch.nn as nn

# 从 YOLO 模型输出的张量中，提取每张图中目标框的最大“置信度”作为损失依据，从而指导 patch 的优化方向
class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """Output must be of the shape [batch, -1, 5 + num_cls]"""
        # get values necessary for transformation
        #确认输出的最后一维是 [5 + 类别数量]，防止错传张量
        assert output.size(-1) == (5 + self.config.n_classes)
        #每一个检测框的输出[batch_size, num_boxes, [x_center, y_center, width, height, objectness_score], class_1_score, class_2_score, ..., class_n_score]
        #提取分类置信度
        class_confs = output[:, :, 5 : 5 + self.config.n_classes]  # [batch, -1, n_classes]
        #提取 objectness 分数
        objectness_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1], no need to run sigmoid here

        if self.config.objective_class_id is not None:
            # norm probs for object classes to [0, 1]
            #原始的class_confs：[batch_size, num_boxes, num_classes]，在最后一维（类别维度）上进行 softmax；把每个框预测的类别分数转换成 概率分布，总和为 1。
            class_confs = torch.nn.Softmax(dim=2)(class_confs)
            # only select the conf score for the objective class
            #对每个框来说，它预测为指定类（目标类）的概率。
            class_confs = class_confs[:, :, self.config.objective_class_id]
            #如果不指定类别，就取最大类别得分
        else:
            # get class with highest conf for each box if objective_class_id is None
            class_confs = torch.max(class_confs, dim=2)[0]  # [batch, -1, 4] -> [batch, -1]
        #计算目标置信度分数（用于损失）
        confs_if_object = self.config.loss_target(objectness_score, class_confs)
        #返回最大置信度（对抗攻击目标）
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf

#鼓励生成的对抗 patch “不那么显眼
class SaliencyLoss(nn.Module):
    """
    Implementation of the colorfulness metric as the saliency loss.

    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        #确保 patch 是 RGB 图像（通道数为 3）才能计算颜色对比
        assert adv_patch.shape[0] == 3
        #拆分颜色通道
        r, g, b = adv_patch
        #构造两个“颜色对比通道”
        rg = r - g
        yb = 0.5 * (r + g) - b
        #构造两个“颜色对比通道”
        mu_rg, sigma_rg = torch.mean(rg) + 1e-8, torch.std(rg) + 1e-8
        mu_yb, sigma_yb = torch.mean(yb) + 1e-8, torch.std(yb) + 1e-8
        #计算整体 saliency 值，第一项：√(σ_rg² + σ_yb²) → 反映 颜色变化的剧烈程度（对比度）第二项：0.3 * √(μ_rg² + μ_yb²) → 表示 是否颜色偏离平均（色偏）
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        #归一化处理
        return sl / torch.numel(adv_patch)

#衡量 adversarial patch 光滑程度 的损失函数
class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        #adv_patch 是 [C, H, W] 的对抗贴图，输出：一个标量 loss，表示这个 patch 的“总变差”
    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W]
        """
        # calc diff in patch rows
        #adv_patch[:, :, 1:] - adv_patch[:, :, :-1]：对于每一行的相邻像素列做差（横向变化）
        tvcomp_r = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0) #把结果压缩成一个标量
        # calc diff in patch columns
        tvcomp_c = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        #加起来就是整个 patch 的总变化程度
        tv = tvcomp_r + tvcomp_c
        # 归一化处理
        return tv / torch.numel(adv_patch)


class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(triplet_scores_fpath, size), requires_grad=False
        )

    def forward(self, adv_patch):
        # calculate euclidean distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, "r", encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack(
                [torch.full((p_h, p_w), r), torch.full((p_h, p_w), g), torch.full((p_h, p_w), b)]
            )
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)
