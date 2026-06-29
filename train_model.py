
import os
import platform
import time
import math
import logging
from datetime import datetime
import json
import copy

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.enable = False
from torch.utils.data import DataLoader
from torchvision import transforms as T

import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator, ValGenerator
from Train_one_epoch import train_one_epoch, validate_one_epoch
from utils import get_memory_info, WeightedDiceBCE
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format


from nets.ACC_UNet import ACC_UNet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.UCTransNet import UCTransNet
from nets.MRDCTUNet import MultiResolutionDCTUNet as MRDCTUNet
from nets.ACC_UNet import ACC_UNet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.SMESwinUnet import SMESwinUnet
from nets.UCTransNet import UCTransNet
from nets.UNet_pp import UNetPP
from nets.EfficientUnet import EfficientUnet, get_efficientunet_b4
from ANI import GABDNet_AHFE
from NAI import GABDNet_IA
from NANI import BDNet_Backbone
from Global import GlobalBDNet
from B8 import FixedBDNetB8
from B16 import FixedBDNetB16
from abd import BDNet_ABDCT
from sobel import BDNet_SobelABDCT
from fixed import BDNet_GaborF
from equal import BDNet_FixedEqualWeights
from prior import BDNet_FixedPriorWeights
from dir2 import BDNet_GaborDir2
from dir6 import BDNet_GaborDir6
from dir8 import BDNet_GaborDir8
from SE import BDNet_SE
from CBAM import BDNet_CBAM
from k5 import BDNet_ABDCTk5
from k7 import BDNet_k7
from nets.CFCM_Net import CFCM_Net
from nets.MobileUViT import MobileUViT
from nets.UAFF import UAFF
from nets.UltraLight_VM_UNet import UltraLight_VM_UNet
from nets.DepthPolyp import DepthPolyp
from nets.MambaLiteUNet import MambaLiteUNet
from nets.MMP_Net import MMP_Net


from GABDNet import GABDNet
from haar import BDNet_Wavelet as haar
from tbdct import TBDCT
from abdct import ABDCT


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


def save_checkpoint(state, save_path, filename=None):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    epoch = state.get('epoch', 0)
    best_model = state.get('best_model', False)
    model_type = state.get('model', 'GABDNet')


    if filename:
        full_filename = os.path.join(save_path, filename)
    elif best_model:
        full_filename = os.path.join(save_path, f'best_model-{model_type}.pth.tar')
    else:
        full_filename = os.path.join(save_path, f'model-{model_type}-{epoch:02d}.pth.tar')


    torch.save(state, full_filename)

    return full_filename


def save_model_checkpoint(epoch, model, optimizer, loss, dice, config_obj, save_dir,
                          best_model=False, model_type='GABDNet', logger=None):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    checkpoint_state = {

        'epoch': epoch,
        'model': model_type,
        'best_model': best_model,
        'timestamp': timestamp,


        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),


        'loss': loss,
        'dice': dice,
        'val_loss': loss,


        'config': {
            'model_name': getattr(config_obj, 'model_name', model_type),
            'task_name': getattr(config_obj, 'task_name', 'Task'),
            'session_name': getattr(config_obj, 'session_name', 'session'),
            'learning_rate': getattr(config_obj, 'learning_rate', 1e-4),
            'batch_size': getattr(config_obj, 'batch_size', 2),
            'img_size': getattr(config_obj, 'img_size', 224),
            'n_channels': getattr(config_obj, 'n_channels', 3),
            'n_labels': getattr(config_obj, 'n_labels', 1),
            'n_filts': getattr(config_obj, 'n_filts', 32),
            'max_epochs': getattr(config_obj, 'max_epochs', 1500),
            'early_stop_patience': getattr(config_obj, 'early_stop_patience', 200),
            'dice_weight': getattr(config_obj, 'dice_weight', 0.7),
            'bce_weight': getattr(config_obj, 'bce_weight', 0.3),
            'gradient_accumulation_steps': getattr(config_obj, 'gradient_accumulation_steps', 1),
        },


        'training_info': {
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }


    saved_path = save_checkpoint(checkpoint_state, save_dir)


    if logger:
        logger.info(f'\t Saving to {saved_path}')
        if best_model:
            logger.info(f'\t Successfully saved best model (Dice: {dice:.4f})')
        else:
            logger.info(f'\t Successfully saved checkpoint for epoch {epoch}')


    if best_model:
        config_file = os.path.join(save_dir, f'training_config-{model_type}.json')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_config': checkpoint_state['config'],
                    'training_info': checkpoint_state['training_info'],
                    'best_performance': {
                        'epoch': epoch,
                        'dice': dice,
                        'loss': loss,
                        'timestamp': timestamp
                    }
                }, f, indent=4, ensure_ascii=False)

            if logger:
                logger.info(f'\t Training configuration saved to {config_file}')
        except Exception as e:
            if logger:
                logger.warning(f'\t Failed to save training configuration: {e}')

    return saved_path


def worker_init_fn(worker_id):

    import numpy as np, random
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def validate_batch_size(user_batch_size, dataset_len):
    if user_batch_size is not None and user_batch_size > 0:
        return min(user_batch_size, dataset_len)
    return min(getattr(config, "batch_size", 2), dataset_len)


def format_memory_info():
    try:
        memory_info = get_memory_info()
        if isinstance(memory_info, dict):

            info_parts = []
            for key, value in memory_info.items():
                info_parts.append(f"{key}: {value}")
            return " | ".join(info_parts)
        elif isinstance(memory_info, str):

            return memory_info
        else:

            return str(memory_info)
    except Exception as e:
        return f"Failed to get memory info: {e}"


def log_model_statistics(model, model_type, logger, input_size):
    try:

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


        try:


            prof_model = copy.deepcopy(model).cpu()
            prof_model.eval()

            dummy_input = torch.randn(input_size)
            with torch.no_grad():
                flops, params = profile(prof_model, inputs=(dummy_input,), verbose=False)

            flops_str, params_str = clever_format([flops, params], "%.3f")


            stats = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'flops': flops,
                'params_str': params_str,
                'flops_str': flops_str,
                'flops_success': True
            }

            logger.info(f"\nModel statistics ({model_type}):")
            logger.info(f"  Total parameters: {params_str}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  FLOPs: {flops_str}")

        except RuntimeError as e:

            error_msg = str(e)
            logger.warning(f"FLOPs calculation failed({type(e).__name__}): {error_msg[:100]}...")


            if model_type == 'ABDCT':
                logger.info(f"  Reason: {model_type}model contains dynamic shape operations(unfold/fold),FLOPs cannot be computed accurately")
            else:
                logger.info(f"  Reason: model contains dynamic shape operations,FLOPs cannot be computed accurately")


            stats = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'flops': None,
                'params_str': f"{total_params / 1e6:.2f}M",
                'flops_str': 'N/A (dynamic calculation)',
                'flops_success': False
            }

            logger.info(f"\nModel statistics ({model_type}):")
            logger.info(f"  Total parameters: {stats['params_str']} ({total_params:,})")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  FLOPs: {stats['flops_str']}")

        except Exception as e:

            logger.warning(f"Unexpected error during FLOPs calculation: {type(e).__name__}: {str(e)[:100]}")

            stats = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'flops': None,
                'params_str': f"{total_params / 1e6:.2f}M",
                'flops_str': 'N/A',
                'flops_success': False
            }

            logger.info(f"\nModel statistics ({model_type}):")
            logger.info(f"  Total parameters: {stats['params_str']} ({total_params:,})")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  FLOPs: {stats['flops_str']}")

        return stats

    except Exception as e:
        logger.warning(f'Failed to record model statistics: {type(e).__name__}: {e}')
        return None


def main_loop(batch_size=None, model_type=None, tensorboard=False):

    log_dir = config.save_path
    os.makedirs(log_dir, exist_ok=True)
    logger = logger_config(os.path.join(log_dir, f"{config.session_name}.log"))


    print(f"Loading dataset: {config.task_name}")

    try:

        train_tf = RandomGenerator(output_size=[config.img_size, config.img_size])
        train_dataset = ImageToImage2D(
            dataset_path=config.train_dataset,
            joint_transform=train_tf,
            image_size=config.img_size,
            n_labels=config.n_labels
        )


        val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
        val_dataset = ImageToImage2D(
            dataset_path=config.val_dataset,
            joint_transform=val_tf,
            image_size=config.img_size,
            n_labels=config.n_labels
        )


        current_batch = validate_batch_size(batch_size, len(train_dataset))

        num_workers = 2

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=(num_workers > 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=current_batch,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(num_workers > 0)
        )

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Training on {platform.uname()[1]}")
        print(f"Using GPU for training: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(format_memory_info())
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision training enabled")
    else:
        device = torch.device('cpu')
        scaler = None
        print("Using CPU for training")


    print(
        f"Initializing {model_type} - input channels: {config.n_channels}, output classes: {config.n_labels}, base filters: {config.n_filts}")
    if model_type in ['SwinUnet', 'SMESwinUnet', 'UCTransNet', 'UNetPP', 'EfficientUnet']:

        if model_type == 'SwinUnet':
            model = SwinUnet(n_labels=config.n_labels, img_size=config.img_size, zero_head=False, vis=False)
            if config.pretrain:
                model.load_from()
        elif model_type == 'UCTransNet':
            config_dict = config.get_CTranS_config()
            config_dict.n_classes = config.n_labels
            model = UCTransNet(config=config_dict, img_size=config.img_size, n_classes=config.n_labels)
        elif model_type == 'UNetPP':
            model = UNetPP(in_channel=config.n_channels, out_channel=config.n_labels)
        elif model_type == 'EfficientUnet':
            model = get_efficientunet_b4(out_channels=config.n_labels, concat_input=True, pretrained=False)
        else:
            model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels, zero_head=False,
                                vis=False)
            if config.pretrain:
                model.load_from()
    else:

        try:
            if model_type == 'MambaLiteUNet':
                model = MambaLiteUNet(
                    num_classes=config.n_labels,
                    input_channels=config.n_channels
                )

            elif model_type in ('GABDNet', 'BDNet'):
                model = GABDNet(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)
            elif model_type == 'MultiResUnet':
                model = eval(model_type)(n_channels=config.n_channels, n_classes=config.n_labels, nfilt=config.n_filts)
            else:
                model = eval(model_type)(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)
        except TypeError as e:
            raise ValueError(f"Model {model_type} initialization failed: {e}. Please check the model constructor arguments.")
    model = model.to(device)
    model.eval()


    lr = getattr(config, "learning_rate", 7e-4)
    print(f"Initial learning rate: {lr}")
    accumulation_steps = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
    effective_batch = current_batch * accumulation_steps
    effective_lr = lr * min(2.0, (effective_batch / max(1, config.batch_size)) ** 0.25)
    print(f"Effective learning rate: {effective_lr}")


    try:

        input_size = (1, config.n_channels, config.img_size, config.img_size)


        model_stats = log_model_statistics(model, model_type, logger, input_size)


        if model_stats:
            stats_file = os.path.join(log_dir, f'{model_type}_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'total_params': model_stats['total_params'],
                    'trainable_params': model_stats['trainable_params'],
                    'flops': model_stats['flops'],
                    'params_formatted': model_stats['params_str'],
                    'flops_formatted': model_stats['flops_str'],
                    'flops_calculation_success': model_stats['flops_success'],
                    'input_size': input_size
                }, f, indent=4)
            logger.info(f'Model statistics saved to: {stats_file}')

    except Exception as e:
        logger.warning(f'Failed to record model statistics: {e}')


    if config.use_adamw:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=effective_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
        print("Using AdamW optimizer")
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=effective_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
        print("Using Adam optimizer")

    dice_weight = getattr(config, "dice_weight", 0.7)
    bce_weight = getattr(config, "bce_weight", 0.3)
    criterion = WeightedDiceBCE(dice_weight=dice_weight, BCE_weight=bce_weight, n_labels=config.n_labels)
    print(f"Loss weights - Dice: {dice_weight}, BCE: {bce_weight}")


    T_0 = max(1, len(train_loader) // 5)
    scheduler = None
    if config.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=lr * 0.001
        )
        print(f"Enabled CosineAnnealingWarmRestarts learning-rate scheduler, T_0: {T_0}")
    else:
        print("Learning-rate scheduler disabled; learning rate is fixed")


    tb_dir = os.path.join(log_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    print(f"TensorBoard log directory: {tb_dir}")

    print("Start training...")
    print("=" * 80)


    best_dice = -1.0
    best_epoch = 1
    epochs = getattr(config, "max_epochs", 3000)
    early_stop_patience = getattr(config, "early_stop_patience", 300)
    early_stop_count = 0


    current_date = datetime.now().strftime("%Y-%m-%d")
    model_save_dir = os.path.join(log_dir, f"models/{current_date}")
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n========= Epoch [{epoch}/{epochs}] =========")


        train_stats = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            logger=logger
        )


        if scheduler is not None:
            scheduler.step(epoch + 1)


        val_stats = validate_one_epoch(
            epoch=epoch,
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            vis_base_path=model_save_dir
        )


        improved = val_stats["dice"] > best_dice
        if improved:
            prev = max(0.0, best_dice)
            best_dice = val_stats["dice"]
            best_epoch = epoch

            print(f"\t Saved best model, mean Dice improved from {prev:.4f} to {best_dice:.4f} (gain: {best_dice - prev:.4f})")


            saved_path = save_model_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loss=val_stats["loss"],
                dice=best_dice,
                config_obj=config,
                save_dir=model_save_dir,
                best_model=True,
                model_type=model_type,
                logger=logger
            )

            early_stop_count = 0
        else:
            early_stop_count += 1
            logger.info(f'\t Mean dice:{val_stats["dice"]:.4f} does not increase, '
                        f'the best is still: {best_dice:.4f} in epoch {best_epoch}')

        print(f"\t Early stopping counter: {early_stop_count}/{early_stop_patience}")


        if torch.cuda.is_available():
            print("\t " + format_memory_info())

        if early_stop_count >= early_stop_patience:
            logger.info("Early stopping triggered; training stopped.")
            break


    logger.info("=" * 50)
    logger.info("Training summary")
    logger.info("=" * 50)
    logger.info(f"Best Dice coefficient: {best_dice:.4f}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Total training epochs: {epoch}")
    logger.info(f"Model save directory: {model_save_dir}")
    logger.info("=" * 50)

    return model


if __name__ == "__main__":
    model = main_loop(
        batch_size=None,
        model_type=getattr(config, "model_name", "GABDNet"),
        tensorboard=True
    )
