
import math
import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import psutil
import scipy.ndimage as ndimage


COMPILE_ENABLED = os.getenv('USE_TORCH_COMPILE', '0') == '1' and torch.__version__ >= '2.0'


def get_memory_info():
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "percent": memory.percent,
        "used": memory.used,
        "free": memory.free
    }


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def compute_dice(pred, gt, smooth=1e-6):
    pred = pred.float()
    gt = gt.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)


    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(pred, size=gt.shape[2:], mode='nearest')


    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > 0.5).float()
    gt_bin = (gt > 0.5).float()


    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    gt_flat = flat(gt_bin)

    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)


    both_empty = (pred_flat.sum(dim=1) == 0) & (gt_flat.sum(dim=1) == 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)


    dice = torch.nan_to_num(dice, nan=0.0, posinf=1.0, neginf=0.0)
    dice = torch.clamp(dice, 0.0, 1.0)

    return dice.mean()


if COMPILE_ENABLED:
    compute_dice = torch.compile(compute_dice)


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()


        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)


        if inputs.shape[2:] != targets.shape[2:]:
            inputs = F.interpolate(inputs, size=targets.shape[2:], mode='nearest')


        inputs_prob = torch.sigmoid(inputs) if inputs.max() > 1.0 or inputs.min() < 0.0 else inputs
        targets_bin = (targets > 0.5).float()


        flat = lambda x: x.view(x.size(0), -1)
        inputs_flat = flat(inputs_prob)
        targets_flat = flat(targets_bin)

        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = torch.where((union == 0), torch.ones_like(dice), dice)
        dice = torch.clamp(dice, 0.0, 1.0)

        return 1.0 - dice.mean()


if COMPILE_ENABLED:
    DiceLoss.forward = torch.compile(DiceLoss.forward)


class WeightedDiceBCE(nn.Module):

    def __init__(self, dice_weight=1, BCE_weight=1, n_labels=1):
        super(WeightedDiceBCE, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.BCE_weight = BCE_weight
        self.n_labels = n_labels

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()


        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)


        if inputs.shape[2:] != targets.shape[2:]:
            inputs = F.interpolate(inputs, size=targets.shape[2:], mode='nearest')


        inputs_prob = torch.sigmoid(inputs)
        targets_bin = (targets > 0.5).float()


        dice = self.dice_loss(inputs_prob, targets_bin)
        bce = self.bce_loss(inputs, targets_bin)

        total_loss = self.dice_weight * dice + self.BCE_weight * bce
        return total_loss


if COMPILE_ENABLED:
    WeightedDiceBCE.forward = torch.compile(WeightedDiceBCE.forward)


class CosineAnnealingWarmRestarts(_LRScheduler):

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        curr_cycle = 0
        curr_iter = self.last_epoch
        while curr_iter >= self.T_0 * (self.T_mult ** curr_cycle):
            curr_iter -= self.T_0 * (self.T_mult ** curr_cycle)
            curr_cycle += 1

        curr_cycle_length = self.T_0 * (self.T_mult ** curr_cycle)

        cycle_progress = curr_iter / max(1, curr_cycle_length)
        cycle_progress = min(1.0, max(0.0, cycle_progress))

        cos_val = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        return [self.eta_min + (base_lr - self.eta_min) * cos_val for base_lr in self.base_lrs]


def save_images(image, mask, pred, save_path, image_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))
    if mask.shape[0] == 1 and len(mask.shape) == 3:
        mask = mask[0]
    if pred.shape[0] == 1 and len(pred.shape) == 3:
        pred = pred[0]

    image = (image * 255).astype(np.uint8)

    if np.max(mask) <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    if np.max(pred) <= 1.0:
        pred = (pred * 255).astype(np.uint8)

    image_path = os.path.join(save_path, f"{image_name}_image.png")
    mask_path = os.path.join(save_path, f"{image_name}_mask.png")
    pred_path = os.path.join(save_path, f"{image_name}_pred.png")

    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(pred_path, pred)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_divide(num, den, default=0.0):
    return num / den if den != 0 else default


def compute_confusion_matrix(pred, gt, threshold=0.5):
    pred = pred.float()
    gt = gt.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)


    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(pred, size=gt.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    gt_bin = (gt > 0.5).float()


    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    gt_flat = flat(gt_bin)

    tp = (pred_flat * gt_flat).sum().item()
    fp = (pred_flat * (1 - gt_flat)).sum().item()
    tn = ((1 - pred_flat) * (1 - gt_flat)).sum().item()
    fn = ((1 - pred_flat) * gt_flat).sum().item()

    return tp, fp, tn, fn


def compute_metrics(pred, gt, threshold=0.5):
    pred = pred.float()
    gt = gt.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)


    if pred.shape[2:] != gt.shape[2:]:
        pred = F.interpolate(pred, size=gt.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    gt_bin = (gt > 0.5).float()


    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    gt_flat = flat(gt_bin)

    tp = (pred_flat * gt_flat).sum().item()
    fp = (pred_flat * (1 - gt_flat)).sum().item()
    tn = ((1 - pred_flat) * (1 - gt_flat)).sum().item()
    fn = ((1 - pred_flat) * gt_flat).sum().item()

    metrics = {}


    metrics['dice'] = safe_divide(2 * tp, 2 * tp + fp + fn)
    metrics['iou'] = safe_divide(tp, tp + fp + fn)
    metrics['accuracy'] = safe_divide(tp + tn, tp + tn + fp + fn)
    metrics['precision'] = safe_divide(tp, tp + fp)
    metrics['recall'] = safe_divide(tp, tp + fn)
    metrics['f1'] = safe_divide(2 * metrics['precision'] * metrics['recall'],
                                metrics['precision'] + metrics['recall'])
    metrics['specificity'] = safe_divide(tn, tn + fp)

    return metrics


def compute_metrics_on_batch(true, pred, threshold=0.5):
    pred = pred.float()
    true = true.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if true.dim() == 3:
        true = true.unsqueeze(1)


    if pred.shape[2:] != true.shape[2:]:
        pred = F.interpolate(pred, size=true.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    true_bin = (true > 0.5).float()

    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    true_flat = flat(true_bin)

    tp = (true_flat * pred_flat).sum(dim=1)
    fp = ((1 - true_flat) * pred_flat).sum(dim=1)
    fn = (true_flat * (1 - pred_flat)).sum(dim=1)
    tn = ((1 - true_flat) * (1 - pred_flat)).sum(dim=1)

    metrics = {
        'precision': safe_divide(tp.sum().item(), (tp + fp).sum().item(), 0.0),
        'recall': safe_divide(tp.sum().item(), (tp + fn).sum().item(), 0.0)
    }


    metrics['f1'] = safe_divide(2 * metrics['precision'] * metrics['recall'],
                                metrics['precision'] + metrics['recall'], 0.0)
    metrics['specificity'] = safe_divide(tn.sum().item(), (tn + fp).sum().item(), 0.0)

    return metrics


def iou_on_batch(true, pred, threshold=0.5):
    pred = pred.float()
    true = true.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if true.dim() == 3:
        true = true.unsqueeze(1)


    if pred.shape[2:] != true.shape[2:]:
        pred = F.interpolate(pred, size=true.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    true_bin = (true > 0.5).float()

    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    true_flat = flat(true_bin)

    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection

    iou = intersection / (union + 1e-6)


    both_empty = (pred_flat.sum(dim=1) == 0) & (true_flat.sum(dim=1) == 0)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)


    iou = torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0)
    iou = torch.clamp(iou, 0.0, 1.0)

    return iou.mean()


if COMPILE_ENABLED:
    iou_on_batch = torch.compile(iou_on_batch)


def pixel_accuracy_on_batch(true, pred, threshold=0.5):
    pred = pred.float()
    true = true.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if true.dim() == 3:
        true = true.unsqueeze(1)


    if pred.shape[2:] != true.shape[2:]:
        pred = F.interpolate(pred, size=true.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    true_bin = (true > 0.5).float()

    flat = lambda x: x.view(x.size(0), -1)
    pred_flat = flat(pred_bin)
    true_flat = flat(true_bin)

    correct = (pred_flat == true_flat).float().sum(dim=1)
    total = pred_flat.size(1)

    acc = correct / max(1, total)


    acc = torch.nan_to_num(acc, nan=0.0, posinf=1.0, neginf=0.0)
    acc = torch.clamp(acc, 0.0, 1.0)

    return acc.mean()


if COMPILE_ENABLED:
    pixel_accuracy_on_batch = torch.compile(pixel_accuracy_on_batch)


def surface_distances(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input must be 2D binary masks")


    dt_a = ndimage.distance_transform_edt(1 - a)
    dt_b = ndimage.distance_transform_edt(1 - b)


    sd_ab = dt_a[b > 0.5]
    sd_ba = dt_b[a > 0.5]

    sd = np.concatenate([sd_ab, sd_ba])
    return sd


def hd95_single(a, b):
    sd = surface_distances(a, b)
    if len(sd) == 0:
        return 0.0
    return np.percentile(sd, 95)


def hd95_on_batch(true, pred, threshold=0.5):
    pred = pred.float()
    true = true.float()


    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if true.dim() == 3:
        true = true.unsqueeze(1)


    if pred.shape[2:] != true.shape[2:]:
        pred = F.interpolate(pred, size=true.shape[2:], mode='nearest')

    pred_prob = torch.sigmoid(pred) if pred.max() > 1.0 or pred.min() < 0.0 else pred
    pred_bin = (pred_prob > threshold).float()
    true_bin = (true > 0.5).float()


    hd95_list = []
    for i in range(pred_bin.size(0)):
        p = pred_bin[i].cpu().numpy()
        t = true_bin[i].cpu().numpy()
        try:
            hd95 = hd95_single(t, p)
            hd95_list.append(hd95)
        except Exception:

            hd95_list.append(0.0)


    avg_hd95 = np.mean(hd95_list)


    avg_hd95 = np.nan_to_num(avg_hd95, nan=0.0, posinf=0.0, neginf=0.0)
    return float(avg_hd95)
