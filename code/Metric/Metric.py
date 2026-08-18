import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import filters

def calculate_psnr_ssim(gt_img, pred_img):
    """
    计算PSNR（峰值信噪比）和SSIM（结构相似性）
    输入：0-1范围的灰度图
    """
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
    ssim = structural_similarity(gt_img, pred_img, data_range=1.0)
    return psnr, ssim

def calculate_vessel_dice(gt_img, pred_img, threshold=0.05):
    """
    计算血管Dice系数：衡量冠脉造影轨迹的重叠度
    用Frangi滤波提取血管掩码，再算Dice
    """
    # Frangi多尺度血管滤波，提取管状结构
    gt_vessel_mask = filters.frangi(gt_img, sigmas=[0.5, 1, 1.5, 2]) > threshold
    pred_vessel_mask = filters.frangi(pred_img, sigmas=[0.5, 1, 1.5, 2]) > threshold

    intersection = np.sum(gt_vessel_mask * pred_vessel_mask)
    union = np.sum(gt_vessel_mask) + np.sum(pred_vessel_mask)
    dice = 2 * intersection / (union + 1e-8)  # 防止除零
    return dice

def evaluate_batch(gt_list, pred_list):
    """批量计算所有指标，返回均值±标准差"""
    psnr_all, ssim_all, dice_all = [], [], []
    for gt, pred in zip(gt_list, pred_list):
        p, s = calculate_psnr_ssim(gt, pred)
        d = calculate_vessel_dice(gt, pred)
        psnr_all.append(p)
        ssim_all.append(s)
        dice_all.append(d)

    return {
        "psnr_mean": np.mean(psnr_all), "psnr_std": np.std(psnr_all),
        "ssim_mean": np.mean(ssim_all), "ssim_std": np.std(ssim_all),
        "dice_mean": np.mean(dice_all), "dice_std": np.std(dice_all)
    }