import os, cv2, argparse, json, math, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur, to_pil_image
from torchvision.utils import save_image

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def total_variation(x):
    dh = torch.abs(x[..., 1:] - x[..., :-1]).mean()
    dv = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    return dh + dv


def clamp_mask_param(mask_param):
    return mask_param.sigmoid()          # range [0, 1]


# --------------------------------------------------
# ------------------ 主流程 ------------------------
# --------------------------------------------------
class BlurMaskTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = select_device(cfg["device"])
        self.model = DetectMultiBackend(cfg["weights_file"], device=self.device)
        self.model.eval().requires_grad_(False)
        names = {k: v.lower() for k, v in self.model.names.items()}
        self.target_cls = [cid for cid, n in names.items() if n in
                           [c.lower() for c in cfg["class_list"]]]
        if not self.target_cls:
            raise ValueError("not ussable class，please check class_list")
        ms = cfg["mask_size"]
        if cfg.get("resume_mask") and Path(cfg["resume_mask"]).is_file():
            self.mask_param = torch.load(cfg["resume_mask"],
                                         map_location=self.device).clone().detach().requires_grad_(True)
            print(f"Resume mask from {cfg['resume_mask']}")
        else:
            self.mask_param_raw = torch.randn(1, 1, ms, ms, device=self.device, requires_grad=True)


        exts = (".jpg", ".jpeg", ".png")
        img_dir = Path(cfg["image_dir"])
        self.img_files = sorted([str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts])
        self.img_files = self.img_files[: cfg["max_train_imgs"]]
        
    def get_clamped_mask(self):
        return self.mask_param_raw.sigmoid()


    # ---------- 把 mask 贴到 bbox 并返回 adversarial image ----------
    def _apply_mask_to_img(self, img0, det_boxes):
        """
        img0: 原始 HxWx3 (BGR) ndarray, range [0,255]
        det_boxes: Tensor[N,4]  xyxy on original scale
        """
        # BGR → RGB → CHW → float
        img = torch.from_numpy(img0[..., ::-1].copy()).permute(2, 0, 1).float().to(self.device) / 255.0
        img_copy = img.clone()
        mask_param = self.get_clamped_mask()

        # 对每个 bbox 贴模糊
        for (x1, y1, x2, y2) in det_boxes.int().cpu().tolist():
            # ---- ROI ----
            region = img[:, y1:y2, x1:x2]              # [3,h,w]
            if region.numel() == 0:                    # 防止空框
                continue
            h, w = region.shape[1:]

            # 1) resize mask, 2) 生成模糊版本, 3) 混合
            m = F.interpolate(mask_param, size=(h, w), mode='bilinear', align_corners=False)[0]  # [1,h,w]
            k = min(h, w) // 3
            k = max(3, k)                  # 最小为 3
            k = k if k % 2 == 1 else k - 1  # 必须是奇数
            k = min(k, self.cfg["blur_kernel"])  # 不超过预设最大值

            if k < 3:
                continue  # 如果区域太小，不模糊

            blurred = gaussian_blur(region.unsqueeze(0), kernel_size=k)[0]
            region_adv = (1 - m) * region + m * blurred
            img_copy[:, y1:y2, x1:x2] = region_adv
        img_np = (img_copy.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_lbox = letterbox(img_np, self.cfg["model_in_sz"], stride=self.model.stride, auto=True)[0]
        img_lbox = img_lbox.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_lbox).float().to(self.device) / 255.0


        return img_tensor  # CHW, float[0,1]

    # ---------- 检测置信度 loss ----------
    def _detection_conf_loss(self, raw_pred):
        """
        raw_pred: Tensor[B, N, 5+C]  (未经 NMS)
        返回目标类别平均置信度
        """
        if isinstance(raw_pred, list):
            raw_pred = raw_pred[0]  # 取出 batch 内第一个 tensor

        if raw_pred.ndim == 3:
            raw_pred = raw_pred[0]   # B==1
        obj = raw_pred[:, 4].sigmoid()                 # objectness
        cls = raw_pred[:, 5:].sigmoid()                # class scores
        # 取目标类别列
        cls_obj = cls[:, self.target_cls] * obj.unsqueeze(1)
        if cls_obj.numel() == 0:
            return torch.tensor(0., device=self.device)
        return cls_obj.mean()

    # ---------- 单 epoch 训练 ----------
    def train_one_epoch(self, optim, epoch):
        pbar = tqdm(self.img_files, desc="Train")
        total_loss, det_loss, tv_loss = 0, 0, 0

        for img_path in pbar:
            # 读取原图并 letterbox 到输入尺寸，先做一次“干净检测”拿 bbox
            img0 = cv2.imread(img_path)
            if img0 is None:
                continue

            img_lbox = letterbox(img0, self.cfg["model_in_sz"],
                                 stride=self.model.stride, auto=True)[0]
            img_lbox = img_lbox.transpose((2, 0, 1))[::-1].copy()
            img_lbox = torch.as_tensor(img_lbox, device=self.device).float() / 255.0
            img_lbox = img_lbox.unsqueeze(0)

            with torch.no_grad():
                det_pred0 = self.model(img_lbox, augment=False, visualize=False)
                det0 = non_max_suppression(det_pred0,
                                           conf_thres=self.cfg["conf_thres"],
                                           iou_thres=self.cfg["iou_thres"],
                                           classes=self.target_cls)[0]

            if det0 is None or len(det0) == 0:
                continue
            # 把 bbox 映射回原图坐标
            det0[:, :4] = scale_boxes(img_lbox.shape[2:], det0[:, :4], img0.shape).round()

            # ---------- 对原图贴 mask 获得 adv_img ----------
            adv_img = self._apply_mask_to_img(img0, det0[:, :4])

            # ---------- forward & loss ----------
            raw_pred = self.model(adv_img.unsqueeze(0), augment=False, visualize=False)
            det_conf = self._detection_conf_loss(raw_pred)

            tv = total_variation(self.get_clamped_mask())
            loss = det_conf + self.cfg["tv_lambda"] * tv

            optim.zero_grad()
            loss.backward()
            optim.step()

            # 日志
            total_loss += loss.item()
            det_loss += det_conf.item()
            tv_loss += tv.item()
            pbar.set_postfix({"det": f"{det_conf:.4f}", "tv": f"{tv:.4f}"})
        if self.cfg.get("save_per_epoch", False):
            sample_img0 = cv2.imread(self.img_files[0])
            sample_lbox = letterbox(sample_img0, self.cfg["model_in_sz"],
                                    stride=self.model.stride, auto=True)[0]
            sample_lbox = sample_lbox.transpose((2, 0, 1))[::-1]
            sample_lbox = sample_lbox.copy()
            sample_lbox = torch.tensor(sample_lbox, device=self.device).float() / 255.0
            sample_lbox = sample_lbox.unsqueeze(0)

            with torch.no_grad():
                det_pred = self.model(sample_lbox)
                det = non_max_suppression(det_pred,
                                        conf_thres=self.cfg["conf_thres"],
                                        iou_thres=self.cfg["iou_thres"],
                                        classes=self.target_cls)[0]
            if det is not None and len(det) > 0:
                det[:, :4] = scale_boxes(sample_lbox.shape[2:], det[:, :4], sample_img0.shape).round()
                adv_img = self._apply_mask_to_img(sample_img0, det[:, :4]).cpu()
                save_dir = Path(self.cfg["save_dir"]) / "train_samples"
                save_dir.mkdir(exist_ok=True)
                save_image(adv_img, save_dir / f"epoch{epoch:02d}.png")
        n = len(self.img_files)
        return total_loss / n, det_loss / n, tv_loss / n

    # ---------- 入口 ----------
    def train(self):
        os.makedirs(self.cfg["save_dir"], exist_ok=True)
        optim = torch.optim.Adam([self.mask_param_raw], lr=self.cfg["lr"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         milestones=self.cfg["lr_milestones"],
                                                         gamma=0.2)

        for epoch in range(1, self.cfg["epochs"] + 1):
            tl, dl, tvl = self.train_one_epoch(optim, epoch)
            scheduler.step()
            print(f"[Epoch {epoch:02d}] loss={tl:.4f}  det={dl:.4f}  tv={tvl:.4f}")

        # 保存 Mask
        mask_save = self.get_clamped_mask().detach().cpu()
        torch.save(mask_save, Path(self.cfg["save_dir"]) / "blur_mask.pt")
        # 保存预览图
        preview = mask_save.expand(3, -1, -1)          # 伪彩
        save_image(preview, Path(self.cfg["save_dir"]) / "blur_mask_preview.png")
        print(f"Mask saved to {self.cfg['save_dir']}")

    # ---------- 推理 / 评估 ----------
    def evaluate(self, out_dir="runs/blurred_eval", max_imgs=20):
        os.makedirs(out_dir, exist_ok=True)
        mask = self.get_clamped_mask()

        for i, img_path in enumerate(tqdm(self.img_files[:max_imgs], desc="Eval")):
            img0 = cv2.imread(img_path)
            if img0 is None:
                continue

            # 先做一次 bbox 检测
            img_lbox = letterbox(img0, self.cfg["model_in_sz"],
                                 stride=self.model.stride, auto=True)[0]
            img_lbox = img_lbox.transpose((2, 0, 1))[::-1].copy()
            img_lbox = torch.as_tensor(img_lbox, device=self.device).float() / 255.0
            img_lbox = img_lbox.unsqueeze(0)
            with torch.no_grad():
                det_pred0 = self.model(img_lbox)
                det0 = non_max_suppression(det_pred0, conf_thres=self.cfg["conf_thres"],
                                           iou_thres=self.cfg["iou_thres"],
                                           classes=self.target_cls)[0]
            if det0 is None or len(det0) == 0:
                continue
            det0[:, :4] = scale_boxes(img_lbox.shape[2:], det0[:, :4], img0.shape).round()

            # 贴 mask
            adv_img = self._apply_mask_to_img(img0, det0[:, :4])
            save_image(adv_img, f"{out_dir}/adv_{i:04d}.png")


# --------------------------------------------------
# ------------------ argparse ----------------------
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON 配置文件")
    parser.add_argument("--eval", action="store_true", help="只做推理/评估")
    return parser.parse_args()


if __name__ == "__main__":
    # ---------------- default cfg -----------------
    cfg = {
        "device": "0",                                   # GPU id
        "weights_file": "runs/train/s_coco_e300_4Class_Vehicle/weights/best.pt",    # 你的 YOLO 权重
        "image_dir": "data/visdrone_data/VisDrone2019-DET-train/images",    # 训练图像目录
        "save_dir": "runs/blur_mask",
        "model_in_sz": [640, 640],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "class_list": ["car", "van", "truck", "bus"],    # 目标类别
        "mask_size": 32,                                 # 初始 mask 分辨率
        "blur_kernel": 15,                               # 高斯核大小 (奇数)
        "tv_lambda": 0.1,                                # TV 正则权重
        "lr": 0.1,
        "lr_milestones": [5, 10],
        "epochs": 15,
        "max_train_imgs": 100000,
        "save_per_epoch": True,
        "resume_mask": None                             
    }

    args = parse_args()
    if args.config:
        cfg.update(json.load(open(args.config)))

    trainer = BlurMaskTrainer(cfg)
    if args.eval:
        trainer.evaluate()
    else:
        trainer.train()
