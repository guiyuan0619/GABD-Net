import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
from tqdm import tqdm
from datetime import datetime
import os


import GABDNet as gabdnet_module
from GABDNet import GABDNet

from haar import BDNet_Wavelet as haar
from tbdct import TBDCT
from abdct import ABDCT
from nets.UNet_base import UNet_base
from nets.ACC_UNet import ACC_UNet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.SMESwinUnet import SMESwinUnet
from nets.UCTransNet import UCTransNet
from nets.UNet_pp import UNetPP
from nets.EfficientUnet import get_efficientunet_b4

from NANI import BDNet_Backbone
from ANI import GABDNet_AHFE
from NAI import GABDNet_IA
from Global import GlobalBDNet
from B8 import FixedBDNetB8
from B16 import FixedBDNetB16
from abd import BDNet_ABDCT
from fixed import BDNet_GaborF
from sobel import BDNet_SobelABDCT
from equal import BDNet_FixedEqualWeights
from prior import BDNet_FixedPriorWeights
from dir2 import BDNet_GaborDir2
from dir6 import BDNet_GaborDir6
from dir8 import BDNet_GaborDir8
from nets.CFCM_Net import CFCM_Net
from nets.UAFF import UAFF
from nets.MobileUViT import MobileUViT
from nets.UltraLight_VM_UNet import UltraLight_VM_UNet
from nets.DepthPolyp import DepthPolyp
from nets.MambaLiteUNet import MambaLiteUNet
from nets.MMP_Net import MMP_Net

from thop import profile, clever_format

import json
from utils import compute_dice, compute_metrics_on_batch, hd95_on_batch
import cv2
import logging
import platform
import torch
torch.backends.cudnn.enabled = False
from sklearn.metrics import jaccard_score
import numpy as np


def format_memory_info():
    if not torch.cuda.is_available():
        return "No GPU available"
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
    return f"Total: {total_memory:.2f} MB, Reserved: {reserved:.2f} MB, Allocated: {allocated:.2f} MB"


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(logging.INFO)
    for h in loggerr.handlers[:]:
        loggerr.removeHandler(h)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    loggerr.addHandler(fh)
    loggerr.addHandler(ch)
    return loggerr


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _add_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {("module." + k): v for k, v in state_dict.items()}


def load_checkpoint(checkpoint_path, model, optimizer=None, logger=None):
    if not os.path.exists(checkpoint_path):
        if logger:
            logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        raw_sd = checkpoint.get('state_dict', checkpoint)

        candidates = []

        inc0 = model.load_state_dict(raw_sd, strict=False)
        candidates.append(("raw", list(inc0.missing_keys), list(inc0.unexpected_keys), raw_sd))

        sd2 = _strip_module_prefix(raw_sd)
        inc2 = model.load_state_dict(sd2, strict=False)
        candidates.append(("strip_module", list(inc2.missing_keys), list(inc2.unexpected_keys), sd2))

        sd3 = _add_module_prefix(raw_sd)
        inc3 = model.load_state_dict(sd3, strict=False)
        candidates.append(("add_module", list(inc3.missing_keys), list(inc3.unexpected_keys), sd3))

        tag, best_missing, best_unexpected, best_sd = min(candidates, key=lambda x: len(x[1]))

        inc_best = model.load_state_dict(best_sd, strict=False)
        best_missing = list(inc_best.missing_keys)
        best_unexpected = list(inc_best.unexpected_keys)

        if logger:
            logger.info(f"Checkpoint state_dict key adaptation strategy: {tag}")
            logger.info(f"missing_keys count: {len(best_missing)}")
            logger.info(f"unexpected_keys count: {len(best_unexpected)}")
            if len(best_missing) > 0:
                logger.warning(f"missing_keys sample (first 20): {best_missing[:20]}")
            if len(best_unexpected) > 0:
                logger.warning(f"unexpected_keys sample (first 20): {best_unexpected[:20]}")

        if len(best_missing) > 10:
            raise RuntimeError(
                f"Incomplete weight loading:missing_keys={len(best_missing)},the current GABDNet structure differs from the training-time structure."
                f"Please confirm that GABDNet.py, or the BDNet.py compatibility wrapper, matches the training-time file; see the startup path print."
            )

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        info = {
            'epoch': checkpoint.get('epoch', -1),
            'dice': checkpoint.get('dice', -1.0),
            'loss': checkpoint.get('loss', float('inf'))
        }
        if logger:
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Epoch: {info['epoch']}, Dice: {info['dice']:.4f}, Loss: {info['loss']:.4f}")
        return info

    except Exception as e:
        if logger:
            logger.error(f"Failed to load checkpoint: {e}")
        return None


def record_model_statistics(model, model_type, input_size=(1, 3, 224, 224), logger=None, save_dir='.', stats_file=None):
    try:
        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print("=" * 50)
        print(f"Model Statistics for {model_type}")
        print("=" * 50)
        print(f"Total Parameters: {params} ({total_params:,})")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Model FLOPs: {flops}")
        print(f"Input Size: {input_size}")
        print("=" * 50)

        if stats_file is None:
            stats_file = os.path.join(save_dir, f"{model_type}_statistics.json")
        stats = {
            "model_type": model_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "flops": flops,
            "input_size": input_size
        }
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        if logger:
            logger.info(f'Model statistics saved to: {stats_file}')

    except Exception as e:
        if logger:
            logger.warning(f'Failed to record model statistics: {e}')


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=dilation)
    boundary = mask - eroded
    boundary = (boundary > 0).astype(np.uint8)
    return boundary


def boundary_f1_score(pred: np.ndarray, gt: np.ndarray, bound_th: int = 2) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred_b = mask_to_boundary(pred)
    gt_b = mask_to_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    k = 2 * bound_th + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    gt_b_dil = cv2.dilate(gt_b, kernel, iterations=1)
    pred_b_dil = cv2.dilate(pred_b, kernel, iterations=1)

    prec = (pred_b & gt_b_dil).sum() / (pred_b.sum() + 1e-6)
    rec = (gt_b & pred_b_dil).sum() / (gt_b.sum() + 1e-6)

    if prec + rec < 1e-6:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return float(f1)


def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)

    a_b = mask_to_boundary(a)
    b_b = mask_to_boundary(b)

    if a_b.sum() == 0 or b_b.sum() == 0:
        return np.array([], dtype=np.float32)

    inv_b = (1 - b_b).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_b, distanceType=cv2.DIST_L2, maskSize=3)
    dists = dist_map[a_b.astype(bool)]
    return dists.astype(np.float32)


def assd_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    d1 = _surface_distances(pred, gt)
    d2 = _surface_distances(gt, pred)

    if d1.size == 0 and d2.size == 0:
        return 0.0
    if d1.size == 0 or d2.size == 0:
        h, w = pred.shape
        return float(np.sqrt(h ** 2 + w ** 2))

    return float((d1.mean() + d2.mean()) / 2.0)


def normalize_to_binary_mask(gt: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt).astype(np.float32)
    if gt.max() > 1.5:
        gt = gt / 255.0
    return (gt > 0.5).astype(np.uint8)


def _debug_print_stats(step_idx, input_tensor, logits, prob, pred_bin, gt_bin):
    x = input_tensor.detach().cpu()
    l = logits.detach().cpu()
    p = prob.detach().cpu()

    x_min, x_max, x_mean = float(x.min()), float(x.max()), float(x.mean())
    l_min, l_max, l_mean = float(l.min()), float(l.max()), float(l.mean())
    p_min, p_max, p_mean = float(p.min()), float(p.max()), float(p.mean())

    pred_pos = float(pred_bin.mean())
    gt_pos = float(gt_bin.mean())

    print(
        f"[DEBUG #{step_idx}] input(min/max/mean)={x_min:.4f}/{x_max:.4f}/{x_mean:.4f} | "
        f"logits(min/max/mean)={l_min:.4f}/{l_max:.4f}/{l_mean:.4f} | "
        f"prob(min/max/mean)={p_min:.4f}/{p_max:.4f}/{p_mean:.4f} | "
        f"pos_ratio(pred/gt)={pred_pos:.6f}/{gt_pos:.6f}"
    )


def vis_and_save_heatmap(model, input_tensor, pred, gt, vis_path, original_filename,
                         dice_pred, dice_ens, debug_step_idx=None):
    gt_bin = normalize_to_binary_mask(gt)

    with torch.no_grad():
        logits = model(input_tensor) if pred is None else pred

    prob = torch.sigmoid(logits)
    pred_bin = (prob > 0.5).cpu().numpy().astype(np.uint8)[0][0]

    if debug_step_idx is not None:
        _debug_print_stats(debug_step_idx, input_tensor, logits, prob, pred_bin, gt_bin)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        dice_t, iou_t, hd95_t, spec_t, bf_t, assd_t = 1.0, 1.0, 0.0, 1.0, 1.0, 0.0
    else:
        gt_t = torch.from_numpy(gt_bin).float().unsqueeze(0).unsqueeze(0)
        pred_t = torch.from_numpy(pred_bin).float().unsqueeze(0).unsqueeze(0)

        dice_t = compute_dice(pred_t, gt_t).item()
        iou_t = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), average='binary')

        try:
            hd95_t = hd95_on_batch(gt_t, pred_t)
        except Exception:
            hd95_t = 0.0

        try:
            metrics = compute_metrics_on_batch(gt_t, pred_t)
            spec_t = float(metrics['specificity'])
        except Exception:
            spec_t = 0.0

        try:
            bf_t = boundary_f1_score(pred_bin, gt_bin, bound_th=2)
        except Exception:
            bf_t = 0.0

        try:
            assd_t = assd_score(pred_bin, gt_bin)
        except Exception:
            assd_t = 0.0

    base_name, ext = os.path.splitext(original_filename)
    pred_filename = f"{base_name}_pred.png"
    gt_filename = original_filename
    cv2.imwrite(os.path.join(vis_path, pred_filename), pred_bin * 255)
    cv2.imwrite(os.path.join(vis_path, gt_filename), gt_bin * 255)

    return float(dice_t), float(iou_t), float(hd95_t), float(spec_t), float(bf_t), float(assd_t)


def main_loop(batch_size=None, model_type='GABDNet', tensorboard=True, test_date=None):
    print("================================================================================\n"
          f"{model_type}test started\n"
          f"Task name: {config.task_name}\n"
          f"Session name: {config.session_name}\n"
          f"Model name: {model_type}\n"
          f"Configured batch size: {config.batch_size}\n"
          "================================================================================")

    print("Loading test dataset...")
    test_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset,
                                  joint_transform=test_tf,
                                  image_size=config.img_size)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size if batch_size is None else batch_size,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             sampler=None,
                             drop_last=False)
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of workers: {2}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {platform.node()}")
    if torch.cuda.is_available():
        print(f"Using GPU for testing: {torch.cuda.get_device_name(0)}")
        print(format_memory_info())

    log_dir = config.save_path
    logger = logger_config(os.path.join(log_dir, f"{config.session_name}.log"))


    print(f"[CHECK] Loaded GABDNet.py path: {gabdnet_module.__file__}")

    print(f"Initializing {model_type} - input channels: {config.n_channels}, output classes: {config.n_labels}, base filters: {config.n_filts}")
    if model_type in ['SwinUnet', 'SMESwinUnet', 'UCTransNet', 'UNetPP', 'EfficientUnet']:
        if model_type == 'SwinUnet':
            model = SwinUnet(n_labels=config.n_labels, img_size=config.img_size, zero_head=False, vis=False)
        elif model_type == 'UCTransNet':
            config_dict = config.get_CTranS_config()
            config_dict.n_classes = config.n_labels
            model = UCTransNet(config=config_dict, img_size=config.img_size, n_classes=config.n_labels)
        elif model_type == 'UNetPP':
            model = UNetPP(in_channel=config.n_channels, out_channel=config.n_labels)
        elif model_type == 'EfficientUnet':
            model = get_efficientunet_b4(out_channels=config.n_labels, concat_input=True, pretrained=False)
        else:
            model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels, zero_head=False, vis=False)
    else:
        try:

            if model_type in ('GABDNet', 'BDNet'):
                model = GABDNet(
                    n_channels=config.n_channels,
                    n_classes=config.n_labels,
                    n_filts=config.n_filts
                )
            elif model_type == 'MultiResUnet':
                model = eval(model_type)(n_channels=config.n_channels, n_classes=config.n_labels, nfilt=config.n_filts)
            else:
                model = eval(model_type)(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)
        except TypeError as e:
            raise ValueError(f"Model {model_type} initialization failed: {e}. Please check the model constructor arguments.")

    model = model.to(device)
    model.eval()

    if test_date is not None:
        current_date = test_date
        print(f"Using specified test date: {current_date}")
    else:
        current_date = datetime.now().strftime("%Y-%m-%d")
        print(f"Using current date: {current_date}")

    checkpoint_dir = os.path.join(config.save_path, f'models/{current_date}')
    best_checkpoint_path = os.path.join(checkpoint_dir, f'best_model-{model_type}.pth.tar')

    load_info = load_checkpoint(best_checkpoint_path, model, logger=logger)
    if load_info is None:
        logger.error("Best model checkpoint was not found or failed to load; test stopped.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    stats_file = os.path.join(log_dir, f"{model_type}_statistics.json")
    record_model_statistics(model, model_type, input_size=(1, config.n_channels, config.img_size, config.img_size),
                            logger=logger, save_dir=log_dir, stats_file=stats_file)

    tb_dir = os.path.join(log_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    print(f"TensorBoard log directory: {tb_dir}")

    print("Start testing...")
    print("=" * 80)

    test_num = len(test_dataset)
    dice_pred = 0.0
    iou_pred = 0.0
    hd95_pred = 0.0
    spec_pred = 0.0
    bf_pred = 0.0
    assd_pred = 0.0
    dice_ens = 0.0

    vis_path = os.path.join(checkpoint_dir, "visualize_test")
    os.makedirs(vis_path, exist_ok=True)

    debug_print_limit = 5

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        dbg_idx = 0
        for batch_idx, sampled_batch in enumerate(test_loader):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']

            for sample_idx in range(test_data.size(0)):
                global_idx = batch_idx * test_loader.batch_size + sample_idx

                try:
                    original_filename = test_dataset.images_list[global_idx]
                except (AttributeError, IndexError):
                    original_filename = f"{global_idx:04d}.png"

                input_img = test_data[sample_idx].unsqueeze(0).to(device)
                lab_single = test_label[sample_idx].squeeze().cpu().numpy()

                debug_step_idx = dbg_idx if dbg_idx < debug_print_limit else None

                dice_pred_t, iou_pred_t, hd95_t, spec_t, bf_t, assd_t = vis_and_save_heatmap(
                    model, input_img, None, lab_single,
                    vis_path, original_filename,
                    dice_pred=dice_pred, dice_ens=dice_ens,
                    debug_step_idx=debug_step_idx
                )

                dbg_idx += 1

                dice_pred += dice_pred_t
                iou_pred += iou_pred_t
                hd95_pred += hd95_t
                spec_pred += spec_t
                bf_pred += bf_t
                assd_pred += assd_t

                torch.cuda.empty_cache()
                pbar.update(1)

    avg_dice = dice_pred / test_num
    avg_iou = iou_pred / test_num
    avg_hd95 = hd95_pred / test_num
    avg_spec = spec_pred / test_num
    avg_bf = bf_pred / test_num
    avg_assd = assd_pred / test_num

    print("dice_pred", avg_dice)
    print("iou_pred", avg_iou)
    print("hd95_pred", avg_hd95)
    print("spec_pred", avg_spec)
    print("bf_score", avg_bf)
    print("assd", avg_assd)

    logger.info("=" * 50)
    logger.info("Test summary")
    logger.info("=" * 50)
    logger.info(f"Mean Dice coefficient: {avg_dice:.4f}")
    logger.info(f"Mean IoU: {avg_iou:.4f}")
    logger.info(f"Mean HD95: {avg_hd95:.4f}")
    logger.info(f"Mean Spec: {avg_spec:.4f}")
    logger.info(f"Mean BF-score: {avg_bf:.4f}")
    logger.info(f"Mean ASSD: {avg_assd:.4f}")
    logger.info(f"Visualization directory: {vis_path}")
    logger.info("=" * 50)

    return avg_dice, avg_iou, avg_hd95, avg_spec, avg_bf, avg_assd


if __name__ == '__main__':
    avg_dice, avg_iou, avg_hd95, avg_spec, avg_bf, avg_assd = main_loop(
        batch_size=None,
        model_type=getattr(config, "model_name", "GABDNet"),
        tensorboard=True,
        test_date='2026-06-05'
    )
