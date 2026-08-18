import os
import cv2
import pydicom
import numpy as np
# from skimage.transform import resize

# =========================================================
# 冠脉CTA
HU_MIN = -200
HU_MAX = 800
WINDOW_LEVEL = 200
WINDOW_WIDTH = 800
# MRI 
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99
# 隔几张处理一张，数值为0则每张图片都处理
SAMPLING_STEP=0
# 模型固定输入尺寸（论文统一256×256）
INPUT_SIZE = 256
# ROI裁剪比例：保留中间75%颅脑区域，裁掉四周空气/颅骨边缘
ROI_RATIO = 0.75

# =========================================================
def get_all_dcm(folder):
    file_list = []
    for root, _, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                try:
                    pydicom.dcmread(full_path, stop_before_pixels=True)
                    file_list.append(full_path)
                    # print(f"识别到DICOM：{full_path}")
                except:
                    continue
    return sorted(file_list)

def hu_to_gray(dcm_file_path):
    """
    CT图像
    单张DICOM转灰度PNG：HU值 -> 0-255灰度
    :param dcm_file_path: 单张.dcm文件路径
    :return: 0-255 uint8灰度图
    """
    dcm = pydicom.dcmread(dcm_file_path)
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    pixel_array = dcm.pixel_array.astype(np.float32)
    hu_array = pixel_array * slope + intercept
    hu_clipped = np.clip(hu_array, HU_MIN, HU_MAX)

    win_min = WINDOW_LEVEL - WINDOW_WIDTH / 2
    win_max = WINDOW_LEVEL + WINDOW_WIDTH / 2
    img_win = np.clip(hu_clipped, win_min, win_max)

    # gray_255 = ((hu_clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    gray_255 = ((img_win - win_min) / (win_max - win_min) * 255).astype(np.uint8)
    return gray_255

def mri_to_gray(dcm_file_path):
    """
    MRI单张DICOM转灰度PNG：自适应百分位截断，保留全信息
    完全对齐DeepGAD原文的直方图强度归一化逻辑
    """    
    dcm = pydicom.dcmread(dcm_file_path)
    pixel_array = dcm.pixel_array.astype(np.float32)
    low = np.percentile(pixel_array, PERCENTILE_LOW)
    high = np.percentile(pixel_array, PERCENTILE_HIGH)
    img_clipped = np.clip(pixel_array, low, high)
    gray_255 = ((img_clipped - low) / (high - low + 1e-8) * 255).astype(np.uint8)
    return gray_255

def get_sorted_dcm(folder):
    """MRI【必须】按Z轴空间位置排序，避免层序错乱"""
    dcm_list = []
    for root, _, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                try:
                    dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                    if hasattr(dcm, "ImagePositionPatient") and dcm.ImagePositionPatient is not None:
                        slice_pos = float(dcm.ImagePositionPatient[2])
                    else:
                        slice_pos = getattr(dcm, 'SliceLocation', 0.0)
                    instance_num = getattr(dcm, 'InstanceNumber', 0)
                    dcm_list.append((slice_pos, instance_num, full_path))
                except:
                    continue
    dcm_list.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in dcm_list]

def crop_brain_roi(img):
    """
    ROI选取：裁剪图像中心75%区域，剔除四周空气、颅骨无效背景
    输入单通道uint8灰度图，返回裁剪后图像
    """
    h, w = img.shape
    # 计算裁剪边界
    crop_h = int(h * ROI_RATIO)
    crop_w = int(w * ROI_RATIO)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    roi_img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
    return roi_img

def sharpen_img(img):
    """USM锐化，强化血管边缘，解决模糊"""
    gauss = cv2.GaussianBlur(img, (0, 0), 1.2)
    sharp = cv2.addWeighted(img, 1.5, gauss, -0.5, 0)
    return sharp

def standardize_image(img):
    # 1 仅做中心ROI裁剪，不做任何降噪、不做锐化
    roi_img = crop_brain_roi(img)
    # 仅仅缩放，INTER_CUBIC保留细节
    resized = cv2.resize(roi_img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    return resized

def dicom_to_png_raw(dcm_root, output_root, sampling_step, select_patient_num, name:str):
    """
    输出结构：
    output_root/
        ├─ A/    (t1n) 平扫
        └─ GT/   (t1c) 增强
    """
    out_A = os.path.join(output_root, "A")
    out_GT = os.path.join(output_root, "GT")
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_GT, exist_ok=True)
    patient_list = sorted([
        p for p in os.listdir(dcm_root)
        if os.path.isdir(os.path.join(dcm_root, p))
    ])

    # 核心改动：截取前N个患者
    if select_patient_num is not None and isinstance(select_patient_num, int):
        patient_list = patient_list[:select_patient_num]
        print(f"【调试模式】仅选取前 {select_patient_num} 位患者进行转换！")

    print(f"共 {len(patient_list)} 名患者，开始转换...")
    for pid in patient_list:
        patient_dir = os.path.join(dcm_root, pid)
        t1n_dir = os.path.join(patient_dir, "ps") #平扫
        # print(t1n_dir)
        t1c_dir = os.path.join(patient_dir, "zq") #增强
        t1n_files = get_sorted_dcm(t1n_dir)
        t1c_files = get_sorted_dcm(t1c_dir)
        assert len(t1n_files) == len(t1c_files), f"{pid} 切片数量不匹配"
        for idx, (f_n, f_c) in enumerate(zip(t1n_files, t1c_files)):
            # if idx % sampling_step != 0:
            #     continue
            img_name = f"{pid}_{idx:04d}.png"
            if name == 'mri':
                img_n = mri_to_gray(f_n)
                img_c = mri_to_gray(f_c)
            elif name == 'ct':
                img_n = hu_to_gray(f_n)
                img_c = hu_to_gray(f_c)

            # # ROI裁剪+降噪+256重采样
            # img_n = standardize_image(img_n)
            # img_c = standardize_image(img_c)
            cv2.imwrite(os.path.join(out_A, img_name), img_n)
            cv2.imwrite(os.path.join(out_GT, img_name), img_c)
        print(f"患者 {pid} 转换完成")
    print("=== DICOM → PNG 全部完成 ===")
    print(f"A: {len(os.listdir(out_A))} 张")
    print(f"GT: {len(os.listdir(out_GT))} 张")



if __name__ == "__main__":  
    RAW_DCM_ROOT = r"E:\data\head_neck_dicom"  # 原始DICOM根目录
    PNG_OUTPUT_ROOT = r"E:\data\head_neck_png4"     # PNG输出根目录
    # dicom_to_png_raw(
    #     dcm_root=RAW_DCM_ROOT,
    #     output_root=PNG_OUTPUT_ROOT,
    #     sampling_step=SAMPLING_STEP,
    #     select_patient_num=20,
    #     name='mri'
    #     )

    dicom_to_png_raw(
            dcm_root=RAW_DCM_ROOT,
            output_root=PNG_OUTPUT_ROOT,
            sampling_step=SAMPLING_STEP,
            select_patient_num=4,
            name='ct'
            )
   
