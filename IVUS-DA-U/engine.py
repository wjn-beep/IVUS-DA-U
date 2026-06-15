import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast
from scipy.ndimage import binary_erosion
from utils import save_imgs


def extract_boundary_points(mask, device=None):
    """
    提取掩码的边界点（基于形态学操作）
    
    Args:
        mask: 二值掩码 (H, W)，torch.Tensor
        device: 计算设备
        
    Returns:
        boundary_points: 边界点坐标 (N, 2)
    """
    if device is None:
        device = mask.device
    
    # 转为numpy进行形态学操作
    mask_np = mask.cpu().numpy()
    
    # 处理空掩码
    if mask_np.sum() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.float32)
    
    # 腐蚀操作
    try:
        eroded = binary_erosion(mask_np, structure=np.ones((3, 3)))
        # 边界 = 原始 - 腐蚀
        boundary = mask_np.astype(bool) & ~eroded
    except:
        # 如果形态学操作失败，返回所有前景点
        boundary = mask_np.astype(bool)
    
    # 提取边界点坐标
    boundary_coords = np.argwhere(boundary)
    
    if len(boundary_coords) == 0:
        # 如果没有边界点（可能是单像素），返回所有非零点
        boundary_coords = np.argwhere(mask_np > 0.5)
    
    # 转回tensor
    boundary_points = torch.from_numpy(boundary_coords).float().to(device)
    
    return boundary_points


def compute_hausdorff_distance_95(pred, target, spacing=None, device=None, batch_size=1000):
    """
    改进的GPU上95% Hausdorff距离计算，基于边界点
    专为IVUS心血管图像设计
    
    Args:
        pred: 预测掩码 (H, W)，二值化，类型为torch.Tensor
        target: 目标掩码 (H, W)，二值化，类型为torch.Tensor
        spacing: 像素间距，如果为None则假设各向同性，间距为1
        device: 计算设备
        batch_size: 批处理大小，控制内存使用
        
    Returns:
        hd95: 95% Hausdorff距离
    """
    if device is None:
        device = pred.device
    
    # 内存预检查
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            memory_free = memory_total - memory_allocated
            
            if memory_free < 1.0:
                batch_size = min(batch_size, 100)
            elif memory_free < 2.0:
                batch_size = min(batch_size, 300)
        except:
            batch_size = 200
    
    # 确保输入是二值掩码
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # 处理空掩码情况
    pred_sum = pred.sum()
    target_sum = target.sum()
    
    if pred_sum == 0 and target_sum == 0:
        return torch.tensor(0.0, device=device)
    elif pred_sum == 0 or target_sum == 0:
        img_diagonal = torch.sqrt(torch.tensor(pred.shape[0]**2 + pred.shape[1]**2, 
                                             device=device, dtype=torch.float32))
        return img_diagonal
    
    # **关键改进：提取边界点而非所有前景点**
    pred_points = extract_boundary_points(pred, device)
    target_points = extract_boundary_points(target, device)
    
    # 检查边界点是否为空
    if len(pred_points) == 0 or len(target_points) == 0:
        img_diagonal = torch.sqrt(torch.tensor(pred.shape[0]**2 + pred.shape[1]**2, 
                                             device=device, dtype=torch.float32))
        return img_diagonal
    
    # 处理边界点过多的情况（使用均匀采样而非随机采样）
    max_points = 5000
    if len(pred_points) > max_points:
        step = len(pred_points) // max_points
        indices = torch.arange(0, len(pred_points), step, device=device)[:max_points]
        pred_points = pred_points[indices]
    if len(target_points) > max_points:
        step = len(target_points) // max_points
        indices = torch.arange(0, len(target_points), step, device=device)[:max_points]
        target_points = target_points[indices]
    
    # 应用间距缩放
    if spacing is None:
        spacing = torch.ones(pred_points.shape[1], device=device)
    else:
        spacing = torch.tensor(spacing, device=device, dtype=torch.float32)
    
    pred_points = pred_points * spacing
    target_points = target_points * spacing
    
    def compute_directed_distance_batch_optimized(source_coords, target_coords, batch_size):
        """优化的批量距离计算"""
        min_distances = []
        current_batch_size = batch_size
        
        for i in range(0, len(source_coords), current_batch_size):
            try:
                batch_source = source_coords[i:i+current_batch_size]
                
                # 内存检查
                if torch.cuda.is_available():
                    memory_free = (torch.cuda.get_device_properties(device).total_memory - 
                                 torch.cuda.memory_allocated(device)) / 1e9
                    if memory_free < 0.5:
                        current_batch_size = max(current_batch_size // 2, 10)
                        if i + current_batch_size > len(source_coords):
                            current_batch_size = len(source_coords) - i
                        batch_source = source_coords[i:i+current_batch_size]
                
                # 计算距离
                batch_source_expanded = batch_source.unsqueeze(1)
                target_expanded = target_coords.unsqueeze(0)
                
                diff = batch_source_expanded - target_expanded
                sq_distances = torch.sum(diff * diff, dim=2)
                
                min_sq_dist = torch.min(sq_distances, dim=1)[0]
                min_dist_batch = torch.sqrt(min_sq_dist + 1e-8)
                
                min_distances.append(min_dist_batch)
                
                del batch_source_expanded, target_expanded, diff, sq_distances, min_sq_dist
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                current_batch_size = max(current_batch_size // 4, 5)
                print(f"HD95计算内存不足，调整批次大小为: {current_batch_size}")
                
                batch_source = source_coords[i:i+current_batch_size]
                try:
                    distances = torch.cdist(batch_source.unsqueeze(0), 
                                          target_coords.unsqueeze(0))[0]
                    min_dist_batch = torch.min(distances, dim=1)[0]
                    min_distances.append(min_dist_batch)
                except:
                    min_dist_batch = []
                    for point in batch_source:
                        point_distances = torch.norm(target_coords - point.unsqueeze(0), dim=1)
                        min_dist_batch.append(torch.min(point_distances))
                    min_distances.append(torch.stack(min_dist_batch))
        
        if min_distances:
            return torch.cat(min_distances)
        else:
            return torch.tensor([0.0], device=device)
    
    try:
        # 计算双向Hausdorff距离
        pred_to_target_distances = compute_directed_distance_batch_optimized(
            pred_points, target_points, batch_size
        )
        
        target_to_pred_distances = compute_directed_distance_batch_optimized(
            target_points, pred_points, batch_size
        )
        
        # 检查结果有效性
        if len(pred_to_target_distances) == 0 or len(target_to_pred_distances) == 0:
            return torch.tensor(0.0, device=device)
        
        # 过滤无效值
        pred_to_target_distances = pred_to_target_distances[torch.isfinite(pred_to_target_distances)]
        target_to_pred_distances = target_to_pred_distances[torch.isfinite(target_to_pred_distances)]
        
        if len(pred_to_target_distances) == 0 or len(target_to_pred_distances) == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算95百分位数
        pred_to_target_95 = torch.quantile(pred_to_target_distances, 0.95)
        target_to_pred_95 = torch.quantile(target_to_pred_distances, 0.95)
        
        # Hausdorff距离是两个方向的最大值
        hd95 = torch.max(pred_to_target_95, target_to_pred_95)
        
        if torch.isfinite(hd95) and hd95 >= 0:
            return hd95
        else:
            return torch.tensor(0.0, device=device)
        
    except Exception as e:
        print(f"HD95计算异常: {e}")
        return torch.tensor(0.0, device=device)
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_optimal_batch_size():
    """智能确定最优批次大小，专为IVUS图像优化"""
    try:
        if not torch.cuda.is_available():
            return 200
        
        current_memory = torch.cuda.memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_memory = total_memory - current_memory
        
        if available_memory > 20:
            return 1000
        elif available_memory > 15:
            return 800
        elif available_memory > 10:
            return 600
        elif available_memory > 5:
            return 400
        elif available_memory > 2:
            return 200
        else:
            return 100
    except:
        return 150


def compute_metrics_gpu(preds, targets, num_classes=2, threshold=0.5, spacing=None):
    """
    优化的GPU多类别指标计算，专为IVUS心血管图像设计
    
    Args:
        preds: 预测结果 (N, C, H, W) 或 (N, H, W)
        targets: 真实标签 (N, H, W)
        num_classes: 类别数
        threshold: 二分类阈值
        spacing: 像素间距，用于HD95计算
    """
    device = preds.device
    epsilon = 1e-8
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    optimal_batch_size = get_optimal_batch_size()
    
    # 二分类
    if num_classes == 2:
        if preds.dim() == 4 and preds.size(1) == 1:
            preds_binary = (preds.squeeze(1) >= threshold).long()
        elif preds.dim() == 3:
            preds_binary = (preds >= threshold).long()
        else:
            preds_binary = torch.argmax(preds, dim=1)
        
        targets_binary = (targets >= 0.5).long()
        
        # 计算混淆矩阵
        pred_flat = preds_binary.view(-1)
        target_flat = targets_binary.view(-1)
        
        # **修复：正确计算混淆矩阵**
        confusion_matrix = torch.zeros(2, 2, dtype=torch.long, device=device)
        for t, p in zip(target_flat, pred_flat):  # ✅ 修复：使用pred_flat
            confusion_matrix[t.long(), p.long()] += 1
            
        tn, fp, fn, tp = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]
        
        # HD95计算
        try:
            hd95 = compute_hausdorff_distance_95(
                preds_binary[0], targets_binary[0], spacing, device, optimal_batch_size
            )
            if not torch.isfinite(hd95):
                hd95 = torch.tensor(0.0, device=device)
        except Exception as e:
            print(f"二分类HD95计算失败: {e}，使用默认值0")
            hd95 = torch.tensor(0.0, device=device)
        
        # 计算指标
        accuracy = (tp + tn).float() / (tp + tn + fp + fn + epsilon)
        precision = tp.float() / (tp + fp + epsilon)
        recall = tp.float() / (tp + fn + epsilon)
        specificity = tn.float() / (tn + fp + epsilon)
        f1 = 2 * tp.float() / (2 * tp + fp + fn + epsilon)
        iou = tp.float() / (tp + fp + fn + epsilon)
        dice = f1
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'sensitivity': recall.item(),
            'specificity': specificity.item(),
            'f1': f1.item(),
            'dice': dice.item(),
            'iou': iou.item(),
            'miou': iou.item(),
            'f1_or_dsc': f1.item(),
            'hd95': hd95.item(),
            'confusion_matrix': confusion_matrix.cpu().numpy()
        }
    
    else:
        # 多类别分类
        if preds.dim() == 4:
            preds_class = torch.argmax(preds, dim=1)
        else:
            preds_class = preds
            
        targets_class = targets.long()
        
        pred_flat = preds_class.view(-1)
        target_flat = targets_class.view(-1)
        
        # 初始化指标
        per_class_iou = torch.zeros(num_classes, device=device)
        per_class_dice = torch.zeros(num_classes, device=device)
        per_class_precision = torch.zeros(num_classes, device=device)
        per_class_recall = torch.zeros(num_classes, device=device)
        per_class_f1 = torch.zeros(num_classes, device=device)
        per_class_hd95 = torch.zeros(num_classes, device=device)
        
        batch_size = preds_class.shape[0]
        
        # 智能采样策略
        sample_indices = list(range(batch_size))
        if batch_size > 10:
            sample_indices = list(range(0, batch_size, max(1, batch_size // 5)))
        
        for class_id in range(num_classes):
            print(f"正在计算类别 {class_id} 的指标...")
            
            # HD95计算
            hd95_values = []
            successful_calculations = 0
            
            for i in sample_indices:
                try:
                    pred_mask = (preds_class[i] == class_id).float()
                    target_mask = (targets_class[i] == class_id).float()
                    
                    if pred_mask.sum() == 0 and target_mask.sum() == 0:
                        hd95_values.append(torch.tensor(0.0, device=device))
                        successful_calculations += 1
                        continue
                    elif pred_mask.sum() == 0 or target_mask.sum() == 0:
                        continue
                    
                    hd95 = compute_hausdorff_distance_95(
                        pred_mask, target_mask, spacing, device, optimal_batch_size
                    )
                    
                    if torch.isfinite(hd95) and hd95 >= 0:
                        hd95_values.append(hd95)
                        successful_calculations += 1
                    
                    if successful_calculations % 3 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"类别 {class_id} 样本 {i} HD95计算失败: {e}")
                    continue
                
                if successful_calculations >= 10:
                    break
            
            # 计算该类别的平均HD95
            if hd95_values:
                per_class_hd95[class_id] = torch.mean(torch.stack(hd95_values))
            else:
                print(f"警告：类别 {class_id} 没有有效的HD95值")
                per_class_hd95[class_id] = torch.tensor(0.0, device=device)
            
            # 计算其他指标
            pred_mask = (pred_flat == class_id)
            target_mask = (target_flat == class_id)
            
            tp = torch.sum(pred_mask & target_mask).float()
            fp = torch.sum(pred_mask & ~target_mask).float()
            fn = torch.sum(~pred_mask & target_mask).float()
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * tp / (2 * tp + fp + fn + epsilon)
            iou = tp / (tp + fp + fn + epsilon)
            dice = f1
            
            per_class_precision[class_id] = precision
            per_class_recall[class_id] = recall
            per_class_f1[class_id] = f1
            per_class_iou[class_id] = iou
            per_class_dice[class_id] = dice
            
            print(f"类别 {class_id} 完成: IoU={iou:.4f}, Dice={dice:.4f}, HD95={per_class_hd95[class_id]:.4f}")
        
        # 计算整体准确率
        accuracy = torch.sum(pred_flat == target_flat).float() / len(pred_flat)
        
        # 计算平均指标
        mean_precision = torch.mean(per_class_precision)
        mean_recall = torch.mean(per_class_recall)
        mean_f1 = torch.mean(per_class_f1)
        mean_iou = torch.mean(per_class_iou)
        mean_dice = torch.mean(per_class_dice)
        mean_hd95 = torch.mean(per_class_hd95)
        
        torch.cuda.empty_cache()
        
        return {
            'accuracy': accuracy.item(),
            'mean_precision': mean_precision.item(),
            'mean_recall': mean_recall.item(),
            'mean_f1': mean_f1.item(),
            'mean_iou': mean_iou.item(),
            'mean_dice': mean_dice.item(),
            'mean_hd95': mean_hd95.item(),
            'miou': mean_iou.item(),
            'f1_or_dsc': mean_f1.item(),
            'per_class_precision': per_class_precision.cpu().numpy(),
            'per_class_recall': per_class_recall.cpu().numpy(),
            'per_class_f1': per_class_f1.cpu().numpy(),
            'per_class_iou': per_class_iou.cpu().numpy(),
            'per_class_dice': per_class_dice.cpu().numpy(),
            'per_class_hd95': per_class_hd95.cpu().numpy()
        }


def train_one_epoch(train_loader,
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
                    return_metrics=False):
    '''
    train model for one epoch
    '''
    model.train() 
 
    loss_list = []
    metrics = {'loss': 0}

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        # 混合精度训练
        if scaler is not None:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if pbar is not None:
            pbar.set_postfix({
                'loss': f'{np.mean(loss_list):.4f}',
                'lr': f'{now_lr:.6f}'
            })
            pbar.update(1)
        else:
            if iter % config.print_interval == 0:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                print(log_info)
                logger.info(log_info)
    
    scheduler.step()
    
    metrics['loss'] = np.mean(loss_list)
    
    if return_metrics:
        return step, metrics
    else:
        return step


def val_one_epoch(test_loader,
                  model,
                  criterion, 
                  epoch, 
                  logger,
                  config,
                  pbar=None,
                  scaler=None,
                  return_metrics=False,
                  force_detailed_metrics=False):
    
    model.eval()
    preds_list = []
    gts_list = []
    loss_list = []
    
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            if scaler is not None:
                with autocast():
                    out = model(img)
                    loss = criterion(out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            if type(out) is tuple:
                out = out[0]
            
            preds_list.append(out)
            gts_list.append(msk)

            if pbar is not None:
                pbar.set_postfix({'val_loss': f'{np.mean(loss_list):.4f}'})
                pbar.update(1)

    all_preds = torch.cat(preds_list, dim=0)
    all_gts = torch.cat(gts_list, dim=0)
    
    metrics = {'loss': np.mean(loss_list)}
    
    if epoch % config.val_interval == 0 or force_detailed_metrics:
        print(f"开始计算详细指标 (Epoch {epoch})...")
        
        torch.cuda.empty_cache()
        
        num_classes = getattr(config, 'num_classes', 2)
        threshold = getattr(config, 'threshold', 0.5)
        spacing = getattr(config, 'spacing', None)
        
        computed_metrics = compute_metrics_gpu(
            all_preds, 
            all_gts.squeeze(1) if all_gts.dim() == 4 else all_gts, 
            num_classes=num_classes, 
            threshold=threshold,
            spacing=spacing
        )
        
        metrics.update(computed_metrics)

        if num_classes > 2 and 'per_class_iou' in metrics:
            class_metrics = {}
            for class_id in range(num_classes):
                class_metrics[class_id] = {
                    'iou': float(metrics['per_class_iou'][class_id]),
                    'dice': float(metrics['per_class_dice'][class_id]),
                    'precision': float(metrics['per_class_precision'][class_id]),
                    'recall': float(metrics['per_class_recall'][class_id]),
                    'f1': float(metrics['per_class_f1'][class_id]),
                    'hd95': float(metrics['per_class_hd95'][class_id])
                }
            metrics['class_metrics'] = class_metrics

        if num_classes == 2:
            log_info = f'val epoch: {epoch}, loss: {metrics["loss"]:.4f}, miou: {metrics["miou"]:.4f}, ' \
                      f'f1_or_dsc: {metrics["f1_or_dsc"]:.4f}, hd95: {metrics["hd95"]:.4f}, ' \
                      f'accuracy: {metrics["accuracy"]:.4f}, specificity: {metrics["specificity"]:.4f}, ' \
                      f'sensitivity: {metrics["sensitivity"]:.4f}, precision: {metrics["precision"]:.4f}, ' \
                      f'dice: {metrics["dice"]:.4f}'
        else:
            log_info = f'val epoch: {epoch}, loss: {metrics["loss"]:.4f}, mean_iou: {metrics["mean_iou"]:.4f}, ' \
                      f'mean_f1: {metrics["mean_f1"]:.4f}, mean_hd95: {metrics["mean_hd95"]:.4f}, ' \
                      f'accuracy: {metrics["accuracy"]:.4f}, mean_precision: {metrics["mean_precision"]:.4f}, ' \
                      f'mean_recall: {metrics["mean_recall"]:.4f}, mean_dice: {metrics["mean_dice"]:.4f}'
        
        if pbar is None:
            print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {metrics["loss"]:.4f}'
        if pbar is None:
            print(log_info)
        logger.info(log_info)
    
    if return_metrics:
        return metrics
    else:
        return metrics['loss']


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None,
                   pbar=None,
                   scaler=None,
                   return_metrics=False,
                   return_class_metrics=False):
    
    model.eval()
    preds_list = []
    gts_list = []
    loss_list = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            if scaler is not None:
                with autocast():
                    out = model(img)
                    loss = criterion(out, msk)
            else:
                out = model(img)
                loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            if type(out) is tuple:
                out = out[0]
            
            preds_list.append(out)
            gts_list.append(msk)
            
            if i % config.save_interval == 0:
                save_imgs(img, msk.cpu().detach().numpy(), out.cpu().detach().numpy(), 
                         i, config.work_dir + 'outputs/', config.datasets, 
                         getattr(config, 'threshold', 0.5), test_data_name=test_data_name)

            if pbar is not None:
                pbar.set_postfix({'test_loss': f'{np.mean(loss_list):.4f}'})
                pbar.update(1)

    all_preds = torch.cat(preds_list, dim=0)
    all_gts = torch.cat(gts_list, dim=0)
    
    torch.cuda.empty_cache()
    
    print("开始计算测试集详细指标...")
    
    num_classes = getattr(config, 'num_classes', 2)
    threshold = getattr(config, 'threshold', 0.5)
    spacing = getattr(config, 'spacing', None)
    
    metrics = compute_metrics_gpu(
        all_preds, 
        all_gts.squeeze(1) if all_gts.dim() == 4 else all_gts, 
        num_classes=num_classes, 
        threshold=threshold,
        spacing=spacing
    )
    
    metrics['loss'] = np.mean(loss_list)

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        if pbar is None:
            print(log_info)
        logger.info(log_info)
    
    class_metrics = None
    if num_classes > 2 and return_class_metrics:
        class_metrics = {}
        for class_id in range(num_classes):
            class_metrics[class_id] = {
                'iou': metrics['per_class_iou'][class_id],
                'dice': metrics['per_class_dice'][class_id],
                'precision': metrics['per_class_precision'][class_id],
                'recall': metrics['per_class_recall'][class_id],
                'f1': metrics['per_class_f1'][class_id],
                'hd95': metrics['per_class_hd95'][class_id]
            }
        
        if pbar is None:
            for class_id, class_metric in class_metrics.items():
                class_log = f'Class {class_id}: precision: {class_metric["precision"]:.4f}, ' \
                           f'recall: {class_metric["recall"]:.4f}, ' \
                           f'f1: {class_metric["f1"]:.4f}, ' \
                           f'iou: {class_metric["iou"]:.4f}, ' \
                           f'dice: {class_metric["dice"]:.4f}, ' \
                           f'hd95: {class_metric["hd95"]:.4f}'
                print(class_log)
                logger.info(class_log)
    
    if num_classes == 2:
        log_info = f'test of best model, loss: {metrics["loss"]:.4f}, miou: {metrics["miou"]:.4f}, ' \
                  f'f1_or_dsc: {metrics["f1_or_dsc"]:.4f}, hd95: {metrics["hd95"]:.4f}, ' \
                  f'accuracy: {metrics["accuracy"]:.4f}, specificity: {metrics["specificity"]:.4f}, ' \
                  f'sensitivity: {metrics["sensitivity"]:.4f}, precision: {metrics["precision"]:.4f}, ' \
                  f'dice: {metrics["dice"]:.4f}'
    else:
        log_info = f'test of best model, loss: {metrics["loss"]:.4f}, mean_iou: {metrics["mean_iou"]:.4f}, ' \
                  f'mean_f1: {metrics["mean_f1"]:.4f}, mean_hd95: {metrics["mean_hd95"]:.4f}, ' \
                  f'accuracy: {metrics["accuracy"]:.4f}, mean_precision: {metrics["mean_precision"]:.4f}, ' \
                  f'mean_recall: {metrics["mean_recall"]:.4f}, mean_dice: {metrics["mean_dice"]:.4f}'
    
    if pbar is None:
        print(log_info)
    logger.info(log_info)

    if return_metrics and return_class_metrics:
        return metrics, class_metrics
    elif return_metrics:
        return metrics
    else:
        return metrics['loss']
