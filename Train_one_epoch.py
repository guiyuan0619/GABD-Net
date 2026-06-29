
import time
import logging
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import Config as config
import os
import cv2
import numpy as np


logger = logging.getLogger("Train_one_epoch")
if not logger.handlers:
    logger.setLevel(logging.INFO)


def _compute_metrics(preds, gts, eps=1e-6):
    if preds.shape != gts.shape:
        preds = F.interpolate(preds, size=gts.shape[-2:], mode='bilinear', align_corners=False)


    if preds.size(1) == 1:

        preds_prob = torch.sigmoid(preds)
    else:

        preds_prob = torch.softmax(preds, dim=1)


    preds_bin = (preds_prob > 0.5).float()
    gts_bin = (gts > 0.5).float()

    inter = (preds_bin * gts_bin).sum(dim=(1, 2, 3))
    union = (preds_bin + gts_bin - preds_bin * gts_bin).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds_bin.sum(dim=(1, 2, 3)) + gts_bin.sum(dim=(1, 2, 3)) + eps)
    acc = (preds_bin.eq(gts_bin)).float().mean(dim=(1, 2, 3))


    from utils import hd95_on_batch
    hd95 = hd95_on_batch(gts_bin, preds_bin)


    from utils import compute_metrics_on_batch
    metrics = compute_metrics_on_batch(gts_bin, preds_bin)
    spec = metrics['specificity']

    return iou.mean().item(), dice.mean().item(), acc.mean().item(), hd95, spec


def train_one_epoch(
        epoch: int,
        model,
        loader,
        optimizer,
        criterion,
        device,
        scaler=None,
        accumulation_steps: int = 1,
        logger: logging.Logger = None
):
    if logger is None:
        logger = logging.getLogger("train")

    model.train()
    num_iters = len(loader)
    loss_avg = 0.0
    iou_avg = 0.0
    dice_avg = 0.0
    acc_avg = 0.0
    hd95_avg = 0.0
    spec_avg = 0.0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    current_batch = loader.batch_size if hasattr(loader, "batch_size") else 1
    effective_batch = current_batch * max(1, accumulation_steps)
    print(f"Current batch size: {current_batch},effective batch size: {effective_batch}")

    for it, batch in enumerate(loader, start=1):
        imgs = batch['image'].to(device, non_blocking=True).float()
        gts = batch['label'].to(device, non_blocking=True).float()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            preds = model(imgs)
            loss = criterion(preds, gts) / max(1, accumulation_steps)


        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()


        need_step = (it % accumulation_steps == 0) or (it == num_iters)
        if need_step:

            clip_grad_norm_(model.parameters(), max_norm=10.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        with torch.no_grad():
            iou, dice, acc, hd95, spec = _compute_metrics(preds, gts)
        loss_avg = (loss_avg * (it - 1) + loss.item() * max(1, accumulation_steps)) / it
        iou_avg = (iou_avg * (it - 1) + iou) / it
        dice_avg = (dice_avg * (it - 1) + dice) / it
        acc_avg = (acc_avg * (it - 1) + acc) / it
        hd95_avg = (hd95_avg * (it - 1) + hd95) / it
        spec_avg = (spec_avg * (it - 1) + spec) / it

        if (it % max(1, num_iters // 40) == 0) or (it == num_iters):

            lr_now = optimizer.param_groups[0]['lr']
            iter_time = time.time() - t0
            print(f"   [Train] Epoch: [{epoch}][{it}/{num_iters}]  "
                  f"Loss:{loss.item() * max(1, accumulation_steps):.3f} (Avg {loss_avg:.4f}) "
                  f"IoU:{iou:.4f} (Avg {iou_avg:.4f}) Dice:{dice:.4f} (Avg {dice_avg:.4f}) "
                  f"Acc:{acc:.4f} (Avg {acc_avg:.4f}) HD95:{hd95:.4f} (Avg {hd95_avg:.4f}) "
                  f"Spec:{spec:.4f} (Avg {spec_avg:.4f}) "
                  f"LR {lr_now:.2e}   "
                  f"Time {iter_time:.1f} (Avg {iter_time / it:.1f})   ")

    logger.info(
        f"Training complete - Loss: {loss_avg:.4f}, Dice: {dice_avg:.4f}, HD95: {hd95_avg:.4f}, Spec: {spec_avg:.4f}")
    return {
        "loss": loss_avg,
        "iou": iou_avg,
        "dice": dice_avg,
        "acc": acc_avg,
        "hd95": hd95_avg,
        "spec": spec_avg
    }


@torch.no_grad()
def validate_one_epoch(
        epoch: int,
        model,
        loader,
        criterion,
        device,
        logger: logging.Logger = None,
        vis_base_path: str = None
):
    if logger is None:
        logger = logging.getLogger("val")

    model.eval()
    num_iters = len(loader)
    loss_avg = 0.0
    iou_avg = 0.0
    dice_avg = 0.0
    acc_avg = 0.0
    hd95_avg = 0.0
    spec_avg = 0.0
    t0 = time.time()

    for it, batch in enumerate(loader, start=1):
        imgs = batch['image'].to(device, non_blocking=True).float()
        gts = batch['label'].to(device, non_blocking=True).float()

        preds = model(imgs)
        loss = criterion(preds, gts)


        if epoch % config.vis_frequency == 0:
            if vis_base_path is None:
                vis_path = config.visualize_path + str(epoch) + '/'
            else:
                vis_path = os.path.join(vis_base_path, 'vis', str(epoch))
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            batch_size = imgs.size(0)
            names = [f"val_{it}_{i}.png" for i in range(batch_size)]
            save_on_batch(imgs, gts, preds, names, vis_path)

        iou, dice, acc, hd95, spec = _compute_metrics(preds, gts)

        loss_avg = (loss_avg * (it - 1) + loss.item()) / it
        iou_avg = (iou_avg * (it - 1) + iou) / it
        dice_avg = (dice_avg * (it - 1) + dice) / it
        acc_avg = (acc_avg * (it - 1) + acc) / it
        hd95_avg = (hd95_avg * (it - 1) + hd95) / it
        spec_avg = (spec_avg * (it - 1) + spec) / it

        if (it % max(1, num_iters // 40) == 0) or (it == num_iters):
            iter_time = time.time() - t0
            print(f"   [Val] Epoch: [{epoch}][{it}/{num_iters}]  "
                  f"Loss:{loss.item():.3f} (Avg {loss_avg:.4f}) "
                  f"IoU:{iou:.4f} (Avg {iou_avg:.4f}) Dice:{dice:.4f} (Avg {dice_avg:.4f}) "
                  f"Acc:{acc:.4f} (Avg {acc_avg:.4f}) HD95:{hd95:.4f} (Avg {hd95_avg:.4f}) "
                  f"Spec:{spec:.4f} (Avg {spec_avg:.4f}) "
                  f"Time {iter_time / it:.1f} (AvgTime {iter_time / it:.1f})   ")

    logger.info(
        f"Validation complete - Loss: {loss_avg:.4f}, Dice: {dice_avg:.4f}, HD95: {hd95_avg:.4f}, Spec: {spec_avg:.4f}")
    return {
        "loss": loss_avg,
        "iou": iou_avg,
        "dice": dice_avg,
        "acc": acc_avg,
        "hd95": hd95_avg,
        "spec": spec_avg
    }


def save_on_batch(imgs, gts, preds, names, save_path):
    batch_size = imgs.size(0)
    for i in range(batch_size):
        pred = torch.sigmoid(preds[i]).cpu().numpy().squeeze()
        pred_bin = (pred > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, names[i]), pred_bin)
