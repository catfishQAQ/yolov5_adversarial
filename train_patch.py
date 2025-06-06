"""
Training code for Adversarial patch training.

python train_patch.py --cfg config_json_file
"""

import glob
import json
import os
import os.path as osp
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from tensorboard import program
from torch import autograd, optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from adv_patch_gen.utils.common import IMG_EXTNS, is_port_in_use, pad_to_square
from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.loss import MaxProbExtractor, NPSLoss, SaliencyLoss, TotalVariationLoss
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from models.common import DetectMultiBackend
from test_patch import PatchTester
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

# optionally set seed for repeatability
SEED = None
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
# setting benchmark to False reduces training time for our setup
torch.backends.cudnn.benchmark = False


class PatchTrainer:
    """Module for training on dataset to generate adv patches."""

    def __init__(self, cfg: edict):
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.model = model.eval()

        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, cfg.mul_gau_mean, cfg.mul_gau_std, cfg.x_off_loc, cfg.y_off_loc, self.dev
        ).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(cfg).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(cfg.triplet_printfile, cfg.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)

        # freeze entire detection model
        for param in self.model.parameters():
            param.requires_grad = False

        # set log dir
        cfg.log_dir = osp.join(cfg.log_dir, f'{time.strftime("%Y%m%d-%H%M%S")}_{cfg.patch_name}')
        #初始化 TensorBoard 写入器; Initialising the TensorBoard Writer
        self.writer = self.init_tensorboard(cfg.log_dir, cfg.tensorboard_port)
        # save config parameters to tensorboard logs
        for cfg_key, cfg_val in cfg.items():
            self.writer.add_text(cfg_key, str(cfg_val))

        # setting train image augmentations
        transforms = None
        if cfg.augment_image:
            transforms = T.Compose(
                [
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),#加入高斯模糊，模拟图像模糊的情况; Add Gaussian blur to simulate a blurred image
                    T.ColorJitter(brightness=0.2, hue=0.04, contrast=0.1),#颜色扰动，包括亮度、色调、对比度的微调; Colour perturbation, including fine tuning of brightness, hue and contrast
                    T.RandomAdjustSharpness(sharpness_factor=2),#随机调整图像的清晰度（锐化效果）; Random adjustment of image sharpness (sharpening effect)
                ]
            )

        # load training dataset
        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(
                image_dir=cfg.image_dir,#图像所在目录; The directory where the image is located
                label_dir=cfg.label_dir,#标签目录，YOLO 格式的 .txt 文件，每张图一个标签文件; Tag directory, .txt file in YOLO format, one tag file per image
                max_labels=cfg.max_labels,#	每张图最多读取多少个标签; Maximum number of tags to read per image
                model_in_sz=cfg.model_in_sz,#模型输入图像大小（H, W）;Model input image size (H, W)
                use_even_odd_images=cfg.use_even_odd_images,#是否只用奇数/偶数编号图片 Whether to number pictures with odd/even numbers only
                transform=transforms,#图像增强 transform; Image enhancement transform
                filter_class_ids=cfg.objective_class_id,#只保留这些目标类别的框; Keep only these target category boxes
                min_pixel_area=cfg.min_pixel_area,#忽略小于该面积的目标框，单位像素;Ignore target boxes smaller than this area, in pixels
                shuffle=True,#是否打乱图像顺序
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.dev.type == "cuda" else False,#如果使用 GPU，将数据固定在内存加快传输
        )
        self.epoch_length = len(self.train_loader)#统计一个 epoch 中包含多少个 batch
    #启动 TensorBoard（一个可视化工具）并创建日志写入器的函数; Functions to start TensorBoard (a visualization tool) and create log writers
    def init_tensorboard(self, log_dir: str = None, port: int = 6006, run_tb=True):
        """Initialize tensorboard with optional name."""
        if run_tb:
            while is_port_in_use(port) and port < 65535:
                port += 1
                print(f"Port {port - 1} is currently in use. Switching to {port} for tensorboard logging")
            #创建一个 TensorBoard 实例；配置它的参数（读取目录 + 端口）；Create a TensorBoard instance; configure its parameters (read directory + port);
            tboard = program.TensorBoard()
            tboard.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
            url = tboard.launch()
            print(f"Tensorboard logger started on {url}")

        if log_dir:
            #返回 SummaryWriter 日志写入器 Returns the SummaryWriter log writer
            return SummaryWriter(log_dir)
        return SummaryWriter()
    #根据指定的类型（灰色 or 随机），生成一个对抗 patch，作为训练优化的起点 Generate an adversarial patch based on the specified type (gray or random) as a starting point for training optimization
    def generate_patch(self, patch_type: str, pil_img_mode: str = "RGB") -> torch.Tensor:
        """
        Generate a random patch as a starting point for optimization.

        Arguments:
            patch_type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
            pil_img_mode: Pillow image modes i.e. RGB, L https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
        """
        #设置 patch 的通道数 Setting the number of channels in a patch
        p_c = 1 if pil_img_mode in {"L"} else 3
        p_w, p_h = self.cfg.patch_size
        #如果类型是 "gray"，生成一个像素值为 0.5 的常量补丁 If the type is "gray", generate a constant patch with a pixel value of 0.5
        if patch_type == "gray":
            adv_patch_cpu = torch.full((p_c, p_h, p_w), 0.5)
        #如果类型是 "random"，生成一个像素值是随机的 patch If the type is "random", generating a pixel value is random patch
        elif patch_type == "random":
            adv_patch_cpu = torch.rand((p_c, p_h, p_w))
        return adv_patch_cpu
    #从指定路径读取一张图片，并将其转换为一个 PyTorch 张量格式的 patch 图像，用于作为对抗补丁的初始值 Reads an image from a specified path and converts it to a patch image in PyTorch tensor format, which is used as the initial value for the adversarial patch
    def read_image(self, path, pil_img_mode: str = "RGB") -> torch.Tensor:
        """
        Read an input image to be used as a patch.

        Arguments:
            path: Path to the image to be read.
        """
        #用 PIL.Image 打开图像文件，.convert(pil_img_mode) 把图像转换成指定的颜色通道格式 Image to open the image file with PIL.Image, .convert(pil_img_mode) to convert the image to the specified color channel format
        patch_img = Image.open(path).convert(pil_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        #把 PIL 图像转换成 PyTorch 的 Tensor 格式，数值范围会变成 [0.0, 1.0]，ToTensor() 会自动把 (H, W, C) 格式转成 (C, H, W) height width channel；
        #Convert the PIL image to PyTorch's Tensor format, the value range will be [0.0, 1.0] and ToTensor() will automatically convert the (H, W, C) format to (C, H, W) height width channel
        adv_patch_cpu = T.ToTensor()(patch_img)
        return adv_patch_cpu

    def train(self) -> None:
        """Optimize a patch to generate an adversarial example."""
        # make output dirs
        patch_dir = osp.join(self.cfg.log_dir, "patches")
        os.makedirs(patch_dir, exist_ok=True)
        if self.cfg.debug_mode:
            for img_dir in ["train_patch_applied_imgs", "val_clean_imgs", "val_patch_applied_imgs"]:
                os.makedirs(osp.join(self.cfg.log_dir, img_dir), exist_ok=True)

        # dump cfg json file
        with open(osp.join(self.cfg.log_dir, "cfg.json"), "w", encoding="utf-8") as json_f:
            json.dump(self.cfg, json_f, ensure_ascii=False, indent=4)

        # fix loss targets
        #配置损失函数目标
        loss_target = self.cfg.loss_target
        if loss_target == "obj":
            self.cfg.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.cfg.loss_target = lambda obj, cls: cls
        elif loss_target in {"obj * cls", "obj*cls"}:
            self.cfg.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(f"Loss target {loss_target} not been implemented")

        # Generate init patch
        supported_modes = {"L", "RGB"}
        if self.cfg.patch_img_mode not in supported_modes:
            raise NotImplementedError(f"Currently only {supported_modes} channels supported")
        if self.cfg.patch_src == "gray":
            adv_patch_cpu = self.generate_patch("gray", self.cfg.patch_img_mode)
        elif self.cfg.patch_src == "random":
            adv_patch_cpu = self.generate_patch("random", self.cfg.patch_img_mode)
        else:
            adv_patch_cpu = self.read_image(self.cfg.patch_src, self.cfg.patch_img_mode)
        #设为可训练 Set to trainable
        adv_patch_cpu.requires_grad = True
        #使用 Adam 优化器，如果 loss 在 50 个 epoch 内没有下降，就降低学习率 Using the Adam optimizer, if the loss does not decrease within 50 epochs, reduce the learning rate
        optimizer = optim.Adam([adv_patch_cpu], lr=self.cfg.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

        start_time = time.time()
        for epoch in range(1, self.cfg.n_epochs + 1):
            #epoch 初始化设置，out_patch_path: 当前轮次保存的 patch 路径，ep_loss: 当前 epoch 总 loss；min_tv_loss: TV loss 的下限；zero_tensor: 用于不启用某项 loss 时作为替代
            #epoch initialization settings, out_patch_path: patch path saved for the current round, ep_loss: total loss for the current epoch, min_tv_loss: lower limit of TV loss, zero_tensor: used as an alternative when a loss is not enabled
            out_patch_path = osp.join(patch_dir, f"e_{epoch}.png")
            ep_loss = 0
            min_tv_loss = torch.tensor(self.cfg.min_tv_loss, device=self.dev)
            zero_tensor = torch.tensor([0], device=self.dev)

            for i_batch, (img_batch, lab_batch) in tqdm(
                enumerate(self.train_loader), desc=f"Running train epoch {epoch}", total=self.epoch_length
            ):
                with autograd.set_detect_anomaly(mode=True if self.cfg.debug_mode else False):
                    img_batch = img_batch.to(self.dev, non_blocking=True)
                    lab_batch = lab_batch.to(self.dev, non_blocking=True)
                    adv_patch = adv_patch_cpu.to(self.dev, non_blocking=True)
                    #开始训练 Start training
                    adv_batch_t = self.patch_transformer(
                        adv_patch,
                        lab_batch,
                        self.cfg.model_in_sz,
                        use_mul_add_gau=self.cfg.use_mul_add_gau,
                        do_transforms=self.cfg.transform_patches,
                        do_rotate=self.cfg.rotate_patches,
                        rand_loc=self.cfg.random_patch_loc,
                    )
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    # resize 到模型输入大小 resize to model input size
                    p_img_batch = F.interpolate(p_img_batch, (self.cfg.model_in_sz[0], self.cfg.model_in_sz[1]))
                    #debug mode: Saving the training images with adversarial patches makes it easier to debug whether the patches are correctly applied, whether the position is reasonable, and whether the visualization effect is normal.
                    if self.cfg.debug_mode:
                        img = p_img_batch[
                            0,
                            :,
                            :,
                        ]
                        img = T.ToPILImage()(img.detach().cpu())
                        img.save(osp.join(self.cfg.log_dir, "train_patch_applied_imgs", f"b_{i_batch}.jpg"))

                    with autocast() if self.cfg.use_amp else nullcontext():
                        #模型预测 + 损失计算 Model Prediction + Loss Calculation
                        output = self.model(p_img_batch)[0]
                        max_prob = self.prob_extractor(output)
                        #计算三个额外的 regularization loss; Compute three additional regularization losses
                        sal = self.sal_loss(adv_patch) if self.cfg.sal_mult != 0 else zero_tensor
                        nps = self.nps_loss(adv_patch) if self.cfg.nps_mult != 0 else zero_tensor
                        tv = self.tv_loss(adv_patch) if self.cfg.tv_mult != 0 else zero_tensor
                    #总 loss + 反向传播 + 更新 Total loss + backpropagation + update
                    det_loss = torch.mean(max_prob)
                    sal_loss = sal * self.cfg.sal_mult
                    nps_loss = nps * self.cfg.nps_mult
                    tv_loss = torch.max(tv * self.cfg.tv_mult, min_tv_loss)

                    loss = det_loss + sal_loss + nps_loss + tv_loss
                    ep_loss += loss
                    #梯度下降优化补丁参数。Gradient descent optimizes the patch parameters.
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # keep patch in cfg image pixel range
                    pl, ph = self.cfg.patch_pixel_range
                    adv_patch_cpu.data.clamp_(pl / 255, ph / 255)
                    #定期记录当前 patch 和各类 loss 到 TensorBoard；Regularly log the current patch and all types of loss to TensorBoard.
                    if i_batch % self.cfg.tensorboard_batch_log_interval == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar("total_loss", loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/det_loss", det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/sal_loss", sal_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/nps_loss", nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("loss/tv_loss", tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar("misc/epoch", epoch, iteration)
                        self.writer.add_scalar("misc/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image("patch", adv_patch_cpu, iteration)
                    if i_batch + 1 < len(self.train_loader):
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss
                        # torch.cuda.empty_cache()  # note emptying cache adds too much overhead
            #每个 epoch 结束，平均 loss 并更新学习率调度器 At the end of each epoch, the loss is averaged and the learning rate scheduler is updated.
            ep_loss = ep_loss / len(self.train_loader)
            scheduler.step(ep_loss)

            # save patch after every patch_save_epoch_freq epochs
            #每隔一定轮数（epoch），把当前优化得到的对抗补丁保存为图片文件，保存在指定的目录下
            #Every certain number of rounds (epoch), save the adversarial patch obtained from the current optimization as an image file in the specified directory
            if epoch % self.cfg.patch_save_epoch_freq == 0:
                img = T.ToPILImage(self.cfg.patch_img_mode)(adv_patch_cpu)
                img.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, sal_loss, nps_loss, tv_loss, loss
                # torch.cuda.empty_cache()  # note emptying cache adds too much overhead

            # run validation to calc asr on val set if self.val_dir is not None
            #每隔固定轮数（val_epoch_freq），在验证集上评估当前的 adversarial patch 攻击效果
            #Every fixed number of rounds (val_epoch_freq), evaluate the effectiveness of the current adversarial patch attack on the validation set
            if all([self.cfg.val_image_dir, self.cfg.val_epoch_freq]) and epoch % self.cfg.val_epoch_freq == 0:
                with torch.no_grad():
                    self.val(epoch, out_patch_path)
        print(f"Total training time {time.time() - start_time:.2f}s")

    def val(self, epoch: int, patchfile: str, conf_thresh: float = 0.4, nms_thresh: float = 0.4) -> None:
        """Calculates the attack success rate according for the patch with respect to different bounding box areas."""
        # load patch from file
        patch_img = Image.open(patchfile).convert(self.cfg.patch_img_mode)
        patch_img = T.Resize(self.cfg.patch_size)(patch_img)
        adv_patch_cpu = T.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)
        #加载验证图片的路径，使用 IMG_EXTNS 限定图片后缀名 Path to load validation image, use IMG_EXTNS to qualify image suffixes
        img_paths = glob.glob(osp.join(self.cfg.val_image_dir, "*"))
        img_paths = sorted([p for p in img_paths if osp.splitext(p)[-1] in IMG_EXTNS])
        #备份 patch 缩放比例，并设置为验证用值 Backup patch scaling and set to validation values
        train_t_size_frac = self.patch_transformer.t_size_frac
        self.patch_transformer.t_size_frac = [0.3, 0.3]  # use a frac of 0.3 for validation
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []

        m_h, m_w = self.cfg.model_in_sz
        cls_id = self.cfg.objective_class_id#从配置文件中读取你要攻击或评估的“目标类别 ID”，并赋值给变量 cls_id；Read the "target class ID" you want to attack or evaluate from the configuration file and assign it to the variable cls_id
        zeros_tensor = torch.zeros([1, 5]).to(self.dev)#没检测到 Not detected 
        #### iterate through all images ####
        for imgfile in tqdm(img_paths, desc=f"Running val epoch {epoch}"):
            img_name = osp.splitext(imgfile)[0].split("/")[-1]
            img = Image.open(imgfile).convert("RGB")
            #使用 pad_to_square() 补成正方形，再 resize 到模型输入大小 Use pad_to_square() to patch to square, then resize to model input size.
            padded_img = pad_to_square(img)
            padded_img = T.Resize(self.cfg.model_in_sz)(padded_img)

            #######################################
            # generate labels to use later for patched image
            padded_img_tensor = T.ToTensor()(padded_img).unsqueeze(0).to(self.dev)
            #用干净图预测目标框，通过 NMS 去除重叠框 Predicting target frames with a clean map and removing overlapping frames by NMS
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            #保存预测得到的“干净标签框” Save the predicted "clean label box".
            all_labels.append(boxes.clone())
            #格式转换成 YOLO 样式的 [class, cx, cy, w, h] Convert the format to YOLO-style [class, cx, cy, w, h].
            boxes = xyxy2xywh(boxes)

            labels = []
            for box in boxes:
                cls_id_box = box[-1].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append([cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])

            # save img if debug mode
            if self.cfg.debug_mode:
                padded_img_drawn = PatchTester.draw_bbox_on_pil_image(all_labels[-1], padded_img, self.cfg.class_list)
                padded_img_drawn.save(osp.join(self.cfg.log_dir, "val_clean_imgs", img_name + ".jpg"))

            # use a filler zeros array for no dets
            #处理无目标情况 Addressing untargeted situations
            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            img_fake_batch = padded_img_tensor
            lab_fake_batch = label.unsqueeze(0).to(self.dev)
            #判断是否有目标，有则贴 patch; Determine if there is a target and patch if there is
            if len(lab_fake_batch[0]) == 1 and torch.equal(lab_fake_batch[0], zeros_tensor):
                # no det, use images without patches
                p_tensor_batch = padded_img_tensor
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch,
                    lab_fake_batch,
                    self.cfg.model_in_sz,
                    use_mul_add_gau=self.cfg.use_mul_add_gau,
                    do_transforms=self.cfg.transform_patches,
                    do_rotate=self.cfg.rotate_patches,
                    rand_loc=self.cfg.random_patch_loc,
                )
                p_tensor_batch = self.patch_applier(img_fake_batch, adv_batch_t)
            #对贴了 patch 的图再次预测目标框 Predict the target frame again for the patched graphs.
            pred = self.model(p_tensor_batch)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
            all_patch_preds.append(boxes.clone())

            # save properly patched img if debug mode
            if self.cfg.debug_mode:
                p_img_pil = T.ToPILImage("RGB")(p_tensor_batch.squeeze(0).cpu())
                p_img_pil_drawn = PatchTester.draw_bbox_on_pil_image(
                    all_patch_preds[-1], p_img_pil, self.cfg.class_list
                )
                p_img_pil_drawn.save(osp.join(self.cfg.log_dir, "val_patch_applied_imgs", img_name + ".jpg"))
        #统计所有预测和目标框 Count all forecasts and target boxes
        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        #计算 ASR（攻击成功率）Calculating ASR (Attack Success Rate)
        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(
            all_labels, all_patch_preds, class_list=self.cfg.class_list, cls_id=cls_id
        )

        print("Validation metrics for images with patches:")
        print(
            f"\tASR@thres={conf_thresh}: asr_s={asr_s:.3f},  asr_m={asr_m:.3f},  asr_l={asr_l:.3f},  asr_a={asr_a:.3f}"
        )

        self.writer.add_scalar("val_asr_per_epoch/area_small", asr_s, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_medium", asr_m, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_large", asr_l, epoch)
        self.writer.add_scalar("val_asr_per_epoch/area_all", asr_a, epoch)
        del adv_batch_t, padded_img_tensor, p_tensor_batch
        torch.cuda.empty_cache()
        self.patch_transformer.t_size_frac = train_t_size_frac


def main():
    parser = get_argparser()
    args = parser.parse_args()
    #load_config_object() 会读取这个 JSON 文件，转成 EasyDict 类型的对象，赋值给 cfg
    #load_config_object() reads the JSON file, turns it into an object of type EasyDict, and assigns it to cfg.
    cfg = load_config_object(args.config)
    trainer = PatchTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
