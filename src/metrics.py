import torch
import numpy as np

def update_depth_metrics(pred, gt, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    valid_pixels = gt > 0
    pred = pred[valid_pixels]
    gt = gt[valid_pixels]
    # pred = pred[valid_pixels].clamp(min=1e-6)
    # gt = gt[valid_pixels].clamp(min=1e-6)
    # n = pred.numel()
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    d1_m.update(d1)
    d2_m.update(d2)
    d3_m.update(d3)
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.float().mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    abs_rel_m.update(abs_rel)
    rmse_m.update(rmse)
    log_10_m.update(log_10)
    rmse_log_m.update(rmse_log)
    silog_m.update(silog)
    sq_rel_m.update(sq_rel)

def compute_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1 = d1_m.compute()
    d2 = d2_m.compute()
    d3 = d3_m.compute()
    abs_rel = abs_rel_m.compute()
    rmse = rmse_m.compute()
    log_10 = log_10_m.compute()
    rmse_log = rmse_log_m.compute()
    silog = silog_m.compute()
    sq_rel = sq_rel_m.compute()
    return d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
    
def reset_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1_m.reset()
    d2_m.reset()
    d3_m.reset()
    abs_rel_m.reset()
    rmse_m.reset()
    log_10_m.reset()
    rmse_log_m.reset()
    silog_m.reset()
    sq_rel_m.reset()

def update_normal_metrics(pred, gt, mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred, gt, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    mean_ae = pred_error.mean()
    median_ae = pred_error.median()
    rmse = torch.sqrt((pred_error ** 2).mean())
    a1 = 100*(pred_error < 5).float().mean()
    a2 = 100*(pred_error < 7.5).float().mean()
    a3 = 100*(pred_error < 11.25).float().mean()
    a4 = 100*(pred_error < 22.5).float().mean()
    a5 = 100*(pred_error < 30).float().mean()
    mean_ae_m.update(mean_ae)
    median_ae_m.update(median_ae)
    rmse_m.update(rmse)
    a1_m.update(a1)
    a2_m.update(a2)
    a3_m.update(a3)
    a4_m.update(a4)
    a5_m.update(a5)

def compute_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae = mean_ae_m.compute()
    median_ae = median_ae_m.compute()
    rmse = rmse_m.compute()
    a1 = a1_m.compute()
    a2 = a2_m.compute()
    a3 = a3_m.compute()
    a4 = a4_m.compute()
    a5 = a5_m.compute()
    return mean_ae, median_ae, rmse, a1, a2, a3, a4, a5

def reset_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae_m.reset()
    median_ae_m.reset()
    rmse_m.reset()
    a1_m.reset()
    a2_m.reset()
    a3_m.reset()
    a4_m.reset()
    a5_m.reset()