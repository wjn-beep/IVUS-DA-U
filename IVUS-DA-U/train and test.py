import os
import csv
import time
import pickle
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.ivus_dataset import IVUSDataset
from models.vmunet.vmunet import VMUNet
from engine import val_one_epoch
from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler, cal_params_flops
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")


CLASS_NAMES = ['Background', 'GH', 'ZZ', 'XW']
TARGET_CLASSES = ['GH', 'ZZ', 'XW']


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask


def check_tensor_validity(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN")
        return False
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf")
        return False
    return True


def apply_gradient_clipping(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def validate_model_outputs(outputs, inputs, loss):
    if not check_tensor_validity(outputs, "model outputs"):
        print("Invalid model outputs detected")
        print(f"Input stats: min={inputs.min():.6f}, max={inputs.max():.6f}, mean={inputs.mean():.6f}")
        return False

    if not check_tensor_validity(loss, "loss"):
        print("Invalid loss detected")
        return False

    return True


def initialize_metrics_csv(config, is_train=True):
    metric_dir = os.path.join(config.work_dir, 'metrics')
    os.makedirs(metric_dir, exist_ok=True)

    filename = 'train_metrics.csv' if is_train else 'val_metrics.csv'
    filepath = os.path.join(metric_dir, filename)

    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'class_name', 'loss', 'dice', 'iou',
                'precision', 'recall', 'hd95', 'learning_rate', 'time_elapsed'
            ])

    return filepath


def append_metrics_to_csv(filepath, epoch, metrics, lr, time_elapsed, class_names):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)

        if 'class_metrics' in metrics:
            for class_id, class_metric in sorted(metrics['class_metrics'].items()):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                writer.writerow([
                    epoch,
                    class_name,
                    metrics.get('loss', 0),
                    class_metric.get('dice', 0),
                    class_metric.get('iou', 0),
                    class_metric.get('precision', 0),
                    class_metric.get('recall', 0),
                    class_metric.get('hd95', 0),
                    lr,
                    time_elapsed
                ])
        elif 'per_class_dice' in metrics:
            for class_id in range(len(class_names)):
                class_name = class_names[class_id]
                writer.writerow([
                    epoch,
                    class_name,
                    metrics.get('loss', 0),
                    metrics['per_class_dice'][class_id] if class_id < len(metrics['per_class_dice']) else 0,
                    metrics['per_class_iou'][class_id] if class_id < len(metrics['per_class_iou']) else 0,
                    metrics.get('per_class_precision', [0] * len(class_names))[class_id] if class_id < len(metrics.get('per_class_precision', [])) else 0,
                    metrics.get('per_class_recall', [0] * len(class_names))[class_id] if class_id < len(metrics.get('per_class_recall', [])) else 0,
                    metrics.get('per_class_hd95', [0] * len(class_names))[class_id] if class_id < len(metrics.get('per_class_hd95', [])) else 0,
                    lr,
                    time_elapsed
                ])


def train_one_epoch_safe(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    step,
    logger,
    config,
    writer,
    pbar=None,
    scaler=None,
    return_metrics=False
):
    model.train()
    loss_list = []
    metrics = {'loss': 0}

    max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
    nan_count = 0
    max_nan_count = 5

    for iter_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float() if torch.cuda.is_available() else images.float()
        targets = targets.cuda(non_blocking=True).long() if torch.cuda.is_available() else targets.long()

        if not check_tensor_validity(images, "input images"):
            logger.warning(f"Skip batch {iter_idx}: invalid input images")
            continue

        try:
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(images)
                    loss = criterion(out, targets)

                if not validate_model_outputs(out, images, loss):
                    nan_count += 1
                    logger.warning(f"Invalid values detected, skip batch {iter_idx} ({nan_count}/{max_nan_count})")
                    if nan_count >= max_nan_count:
                        raise ValueError("Too many invalid batches during training")
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()

                if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                       for p in model.parameters() if p.grad is not None):
                    logger.warning(f"Invalid gradients detected, skip batch {iter_idx}")
                    optimizer.zero_grad()
                    continue

                scaler.unscale_(optimizer)
                apply_gradient_clipping(model, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            else:
                out = model(images)
                loss = criterion(out, targets)

                if not validate_model_outputs(out, images, loss):
                    nan_count += 1
                    logger.warning(f"Invalid values detected, skip batch {iter_idx} ({nan_count}/{max_nan_count})")
                    if nan_count >= max_nan_count:
                        raise ValueError("Too many invalid batches during training")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                       for p in model.parameters() if p.grad is not None):
                    logger.warning(f"Invalid gradients detected, skip batch {iter_idx}")
                    optimizer.zero_grad()
                    continue

                apply_gradient_clipping(model, max_grad_norm)
                optimizer.step()

            nan_count = 0
            step += 1
            loss_list.append(loss.item())
            now_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/loss', loss.item(), global_step=step)

            if pbar is not None:
                pbar.set_postfix({
                    'loss': f'{np.mean(loss_list):.4f}',
                    'lr': f'{now_lr:.6f}'
                })
                pbar.update(1)

        except Exception as e:
            logger.error(f"Error in training batch {iter_idx}: {str(e)}")
            optimizer.zero_grad()
            continue

    scheduler.step()
    metrics['loss'] = np.mean(loss_list) if loss_list else float('inf')

    if return_metrics:
        return step, metrics
    return step


def compute_target_metrics(metrics, class_names, target_classes):
    target_class_indices = [class_names.index(name) for name in target_classes]

    target_dice_scores = []
    target_iou_scores = []
    target_hd95_scores = []
    target_precision_scores = []
    target_recall_scores = []

    if 'class_metrics' in metrics:
        for class_id in target_class_indices:
            if class_id in metrics['class_metrics']:
                target_dice_scores.append(metrics['class_metrics'][class_id].get('dice', 0.0))
                target_iou_scores.append(metrics['class_metrics'][class_id].get('iou', 0.0))
                target_hd95_scores.append(metrics['class_metrics'][class_id].get('hd95', 0.0))
                target_precision_scores.append(metrics['class_metrics'][class_id].get('precision', 0.0))
                target_recall_scores.append(metrics['class_metrics'][class_id].get('recall', 0.0))

    elif 'per_class_dice' in metrics:
        for class_id in target_class_indices:
            if class_id < len(metrics['per_class_dice']):
                target_dice_scores.append(metrics['per_class_dice'][class_id])
                target_iou_scores.append(metrics['per_class_iou'][class_id] if class_id < len(metrics['per_class_iou']) else 0.0)
                target_hd95_scores.append(metrics['per_class_hd95'][class_id] if 'per_class_hd95' in metrics and class_id < len(metrics['per_class_hd95']) else 0.0)
                target_precision_scores.append(metrics['per_class_precision'][class_id] if 'per_class_precision' in metrics and class_id < len(metrics['per_class_precision']) else 0.0)
                target_recall_scores.append(metrics['per_class_recall'][class_id] if 'per_class_recall' in metrics and class_id < len(metrics['per_class_recall']) else 0.0)

    return {
        'target_dice_scores': target_dice_scores,
        'target_iou_scores': target_iou_scores,
        'target_hd95_scores': target_hd95_scores,
        'target_precision_scores': target_precision_scores,
        'target_recall_scores': target_recall_scores,
        'target_dice': np.mean(target_dice_scores) if target_dice_scores else 0.0,
        'target_iou': np.mean(target_iou_scores) if target_iou_scores else 0.0,
        'target_hd95': np.mean(target_hd95_scores) if target_hd95_scores else 0.0,
        'target_precision': np.mean(target_precision_scores) if target_precision_scores else 0.0,
        'target_recall': np.mean(target_recall_scores) if target_recall_scores else 0.0
    }


def log_class_metrics(logger, metrics, class_names, title="Validation Metrics"):
    logger.info(f"\n{'=' * 100}")
    logger.info(title)
    logger.info(f"{'=' * 100}")
    logger.info(f"{'Class':<12} {'Dice':<10} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'HD95':<10} {'Samples':<10}")
    logger.info("-" * 100)

    if 'class_metrics' in metrics:
        for class_id, metric in sorted(metrics['class_metrics'].items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            logger.info(
                f"{class_name:<12} "
                f"{metric.get('dice', 0.0):<10.4f} "
                f"{metric.get('iou', 0.0):<10.4f} "
                f"{metric.get('precision', 0.0):<10.4f} "
                f"{metric.get('recall', 0.0):<10.4f} "
                f"{metric.get('hd95', 0.0):<10.4f} "
                f"{metric.get('samples', 0):<10d}"
            )
    elif 'per_class_dice' in metrics:
        for class_id in range(len(class_names)):
            logger.info(
                f"{class_names[class_id]:<12} "
                f"{metrics['per_class_dice'][class_id] if class_id < len(metrics['per_class_dice']) else 0:<10.4f} "
                f"{metrics['per_class_iou'][class_id] if class_id < len(metrics['per_class_iou']) else 0:<10.4f} "
                f"{metrics.get('per_class_precision', [0]*len(class_names))[class_id] if class_id < len(metrics.get('per_class_precision', [])) else 0:<10.4f} "
                f"{metrics.get('per_class_recall', [0]*len(class_names))[class_id] if class_id < len(metrics.get('per_class_recall', [])) else 0:<10.4f} "
                f"{metrics.get('per_class_hd95', [0]*len(class_names))[class_id] if class_id < len(metrics.get('per_class_hd95', [])) else 0:<10.4f} "
                f"{'N/A':<10}"
            )

    logger.info("=" * 100 + "\n")


def evaluate_best_model_on_test_set(model, test_loader, criterion, config, logger):
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating best model on test set...")
    logger.info("=" * 80)

    test_pbar = tqdm(total=len(test_loader), desc="Test", leave=True)

    test_metrics = val_one_epoch(
        test_loader,
        model,
        criterion,
        epoch=0,
        logger=logger,
        config=config,
        pbar=test_pbar,
        scaler=None,
        return_metrics=True,
        force_detailed_metrics=True
    )

    test_pbar.close()

    test_csv_path = os.path.join(config.work_dir, 'metrics', 'test_metrics.csv')
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)

    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'dice', 'iou', 'precision', 'recall', 'hd95', 'samples'])

        if 'class_metrics' in test_metrics:
            for class_id, metric in sorted(test_metrics['class_metrics'].items()):
                class_name = config.class_names[class_id] if class_id < len(config.class_names) else f"Class_{class_id}"
                writer.writerow([
                    class_name,
                    metric.get('dice', 0),
                    metric.get('iou', 0),
                    metric.get('precision', 0),
                    metric.get('recall', 0),
                    metric.get('hd95', 0),
                    metric.get('samples', 0)
                ])

    log_class_metrics(logger, test_metrics, config.class_names, title="Test Metrics")

    target_metrics = compute_target_metrics(test_metrics, config.class_names, TARGET_CLASSES)
    logger.info("Target classes (GH/ZZ/XW) on test set:")
    logger.info(f"  Mean Dice: {target_metrics['target_dice']:.4f}")
    logger.info(f"  Mean IoU: {target_metrics['target_iou']:.4f}")
    logger.info(f"  Mean HD95: {target_metrics['target_hd95']:.4f}")
    logger.info(f"  Mean Precision: {target_metrics['target_precision']:.4f}")
    logger.info(f"  Mean Recall: {target_metrics['target_recall']:.4f}")

    return test_metrics


def main(config):
    config.class_names = CLASS_NAMES
    config.num_classes = 4
    config.model_config['num_classes'] = 4
    config.max_grad_norm = 1.0

    patience = 30
    patience_counter = 0
    best_target_dice = 0.0
    best_epoch = 0
    best_val_metrics = None
    final_epoch = 0
    test_metrics = None

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)

    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available, training on CPU")
        config.use_mixed_precision = False

    if config.use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    print('#----------Preparing dataset----------#')
    logger.info(f"Class names: {config.class_names}")

    train_dataset_raw = IVUSDataset(
        img_dir=os.path.join(config.data_path, 'train', 'images'),
        mask_dir=os.path.join(config.data_path, 'train', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=4,
        mask_folder_names=['GH', 'ZZ', 'XW'],
        transform=None
    )

    val_dataset_raw = IVUSDataset(
        img_dir=os.path.join(config.data_path, 'val', 'images'),
        mask_dir=os.path.join(config.data_path, 'val', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=4,
        mask_folder_names=['GH', 'ZZ', 'XW'],
        transform=None
    )

    test_dataset_raw = IVUSDataset(
        img_dir=os.path.join(config.data_path, 'test', 'images'),
        mask_dir=os.path.join(config.data_path, 'test', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=4,
        mask_folder_names=['GH', 'ZZ', 'XW'],
        transform=None
    )

    train_dataset = TransformDataset(train_dataset_raw, config.train_transformer)
    val_dataset = TransformDataset(val_dataset_raw, config.test_transformer)
    test_dataset = TransformDataset(test_dataset_raw, config.test_transformer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    logger.info(
        f"Dataset size -> Train: {len(train_dataset_raw)}, "
        f"Val: {len(val_dataset_raw)}, Test: {len(test_dataset_raw)}"
    )

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config

    if config.network != 'vmunet':
        raise ValueError("Only 'vmunet' is supported in this open-source version.")

    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()

    if torch.cuda.is_available():
        model = model.cuda()

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    cal_params_flops(model, config.input_size_h, logger)

    print('#----------Preparing loss, optimizer and scheduler----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    train_metrics_path = initialize_metrics_csv(config, is_train=True)
    val_metrics_path = initialize_metrics_csv(config, is_train=False)

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    start_epoch = 1
    step = 0

    if os.path.exists(resume_model):
        logger.info(f"Resuming from checkpoint: {resume_model}")
        checkpoint = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_target_dice = checkpoint.get('best_target_dice', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        best_val_metrics = checkpoint.get('best_val_metrics', None)

    logger.info(f"Training for {config.epochs} epochs")
    logger.info("Best model selection criterion: mean Dice of GH/ZZ/XW on validation set")

    for epoch in range(start_epoch, config.epochs + 1):
        final_epoch = epoch
        epoch_start = time.time()

        logger.info(f"\n===== Epoch {epoch}/{config.epochs} =====")

        train_pbar = tqdm(total=len(train_loader), desc=f"Train {epoch}", leave=True)
        train_start = time.time()

        step, train_metrics = train_one_epoch_safe(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            pbar=train_pbar,
            scaler=scaler,
            return_metrics=True
        )

        train_pbar.close()
        train_time = time.time() - train_start

        current_lr = optimizer.param_groups[0]['lr']
        append_metrics_to_csv(train_metrics_path, epoch, train_metrics, current_lr, train_time, config.class_names)

        val_pbar = tqdm(total=len(val_loader), desc=f"Val {epoch}", leave=True)
        val_start = time.time()

        val_metrics = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            pbar=val_pbar,
            scaler=scaler,
            return_metrics=True,
            force_detailed_metrics=True
        )

        val_pbar.close()
        val_time = time.time() - val_start

        target_metrics = compute_target_metrics(val_metrics, config.class_names, TARGET_CLASSES)
        val_metrics.update(target_metrics)

        append_metrics_to_csv(val_metrics_path, epoch, val_metrics, current_lr, val_time, config.class_names)
        log_class_metrics(logger, val_metrics, config.class_names, title=f"Validation Metrics - Epoch {epoch}")

        logger.info("Target classes (GH/ZZ/XW):")
        logger.info(f"  GH Dice: {target_metrics['target_dice_scores'][0]:.4f}" if len(target_metrics['target_dice_scores']) > 0 else "  GH Dice: N/A")
        logger.info(f"  ZZ Dice: {target_metrics['target_dice_scores'][1]:.4f}" if len(target_metrics['target_dice_scores']) > 1 else "  ZZ Dice: N/A")
        logger.info(f"  XW Dice: {target_metrics['target_dice_scores'][2]:.4f}" if len(target_metrics['target_dice_scores']) > 2 else "  XW Dice: N/A")
        logger.info(f"  Mean Dice: {target_metrics['target_dice']:.4f}")
        logger.info(f"  Mean IoU: {target_metrics['target_iou']:.4f}")
        logger.info(f"  Mean HD95: {target_metrics['target_hd95']:.4f}")

        if target_metrics['target_dice'] > best_target_dice:
            best_target_dice = target_metrics['target_dice']
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0

            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            with open(os.path.join(checkpoint_dir, 'best_metrics.pkl'), 'wb') as f:
                pickle.dump(best_val_metrics, f)

            logger.info(f"New best model found at epoch {epoch}, mean target Dice = {best_target_dice:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        save_dict = {
            'epoch': epoch,
            'best_target_dice': best_target_dice,
            'best_epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_metrics': best_val_metrics,
            'patience_counter': patience_counter
        }

        if scaler is not None:
            save_dict['scaler_state_dict'] = scaler.state_dict()

        torch.save(save_dict, os.path.join(checkpoint_dir, 'latest.pth'))

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.2f}s")
        logger.info(f"Best mean Dice (GH/ZZ/XW): {best_target_dice:.4f} at epoch {best_epoch}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training completed.")

    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        best_state = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(best_state)
        if torch.cuda.is_available():
            model = model.cuda()

        test_metrics = evaluate_best_model_on_test_set(model, test_loader, criterion, config, logger)
    else:
        logger.warning("Best model not found. Skip test evaluation.")

    writer.close()

    print("\n" + "=" * 60)
    print("Training summary")
    print("=" * 60)
    print(f"Finished epoch: {final_epoch}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best mean Dice of GH/ZZ/XW: {best_target_dice:.4f}")
    if test_metrics is not None:
        test_target = compute_target_metrics(test_metrics, config.class_names, TARGET_CLASSES)
        print(f"Test mean Dice of GH/ZZ/XW: {test_target['target_dice']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    config = setting_config
    config.use_mixed_precision = False
    main(config)
