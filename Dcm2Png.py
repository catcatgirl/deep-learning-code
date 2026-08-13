import os
import cv2
import shutil
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split

# ===================== ====================================
# 冠脉CTA
HU_MIN = -200
HU_MAX = 800

# MRI 
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99

#数据划分比例
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SAMPLING_STEP=0
# =========================================================

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
    gray_255 = ((hu_clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return gray_255

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
    """MRI【必须】按Z轴空间位置排序，避免层序错乱（MRI也必须做）"""
    dcm_list = []
    for root, _, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if os.path.isfile(full_path):
                try:
                    dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
                    slice_pos = getattr(dcm, 'SliceLocation', 0.0)
                    instance_num = getattr(dcm, 'InstanceNumber', 0)
                    dcm_list.append((slice_pos, instance_num, full_path))
                except:
                    continue
    dcm_list.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in dcm_list]

def dicom_to_png_raw(dcm_root, output_root, sampling_step, select_patient_num):
    """
    输出结构：
    output_root/
        ├─ A/    (t1n)
        └─ GT/   (t1c)
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
        t1n_dir = os.path.join(patient_dir, "t1n")
        t1c_dir = os.path.join(patient_dir, "t1c")
        t1n_files = get_sorted_dcm(t1n_dir)
        t1c_files = get_sorted_dcm(t1c_dir)
        assert len(t1n_files) == len(t1c_files), f"{pid} 切片数量不匹配"
        for idx, (f_n, f_c) in enumerate(zip(t1n_files, t1c_files)):
            # if idx % sampling_step != 0:
            #     continue
            img_name = f"{pid}_{idx:04d}.png"
            img_n = mri_to_gray(f_n)
            img_c = mri_to_gray(f_c)
            cv2.imwrite(os.path.join(out_A, img_name), img_n)
            cv2.imwrite(os.path.join(out_GT, img_name), img_c)
        print(f"患者 {pid} 转换完成")
    print("=== DICOM → PNG 全部完成 ===")
    print(f"A: {len(os.listdir(out_A))} 张")
    print(f"GT: {len(os.listdir(out_GT))} 张")

def filter_black_slices(png_root, min_avg_gray, enable_backup):
    '''
    min_avg_gray = 8       # 平均灰度低于这个值视为全黑切片，可微调
    enable_backup = True   # 删之前先备份到backup文件夹，防止误删
    '''
    path_A = os.path.join(png_root, "A")
    path_GT = os.path.join(png_root, "GT")
     # 备份目录
    if enable_backup:
        backup_A = os.path.join(png_root, "backup_black", "A")
        backup_GT = os.path.join(png_root, "backup_black", "GT")
        os.makedirs(backup_A, exist_ok=True)
        os.makedirs(backup_GT, exist_ok=True)
    # 获取所有配对的文件名
    img_files = sorted([f for f in os.listdir(path_A) if f.endswith(".png")])
    total_num = len(img_files)
    remove_num = 0
    valid_files = []
    print(f"\n开始过滤全黑切片，总切片数：{total_num}")
    for fname in img_files:
        img_A_path = os.path.join(path_A, fname)
        img_GT_path = os.path.join(path_GT, fname)
        
        # 两张图都存在才判断
        if not os.path.exists(img_GT_path):
            continue
        
        # 读取灰度图计算平均亮度
        img_A = cv2.imread(img_A_path, cv2.IMREAD_GRAYSCALE)
        img_GT = cv2.imread(img_GT_path, cv2.IMREAD_GRAYSCALE)
        
        avg_A = np.mean(img_A)
        avg_GT = np.mean(img_GT)
        # 只要有一张是全黑，就过滤掉
        if avg_A < min_avg_gray or avg_GT < min_avg_gray:
            remove_num += 1
            # 备份或直接删除
            if enable_backup:
                shutil.move(img_A_path, os.path.join(backup_A, fname))
                shutil.move(img_GT_path, os.path.join(backup_GT, fname))
            else:
                os.remove(img_A_path)
                os.remove(img_GT_path)
        else:
            valid_files.append(fname)
    
    print(f"过滤完成：删除全黑切片 {remove_num} 张，剩余有效切片 {len(valid_files)} 张")
    if enable_backup:
        print(f"被删除的切片已备份到：{os.path.join(png_root, 'backup_black')}")
    return valid_files


def split_dataset_by_patient(png_root, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    从 A / GT 中按患者ID拆分数据集
    """
    path_A = os.path.join(png_root, "A")
    path_GT = os.path.join(png_root, "GT")
    # 从文件名提取患者ID
    img_files = [f for f in os.listdir(path_A) if f.endswith(".png")]
    patient_ids = sorted(list({f.split("_")[0] for f in img_files}))
    print(f"共 {len(patient_ids)} 名患者，开始划分数据集")
    # 划分患者
    train_val_pids, test_pids = train_test_split(patient_ids, test_size=test_ratio, random_state=random_state)
    train_pids, val_pids = train_test_split(train_val_pids, test_size=val_ratio/(1-test_ratio), random_state=random_state)
    print(f"训练患者: {len(train_pids)}")
    print(f"验证患者: {len(val_pids)}")
    print(f"测试患者: {len(test_pids)}")
    # 创建目录
    subsets = [
        ("train", train_pids),
        ("val", val_pids),
        ("test", test_pids)
    ]

    for subset_name, pids in subsets:
        A_dir = os.path.join(png_root, f"{subset_name}A")
        GT_dir = os.path.join(png_root, f"{subset_name}GT")
        os.makedirs(A_dir, exist_ok=True)
        os.makedirs(GT_dir, exist_ok=True)
        for pid in pids:
            for fname in img_files:
                if fname.startswith(pid + "_"):
                    src_A = os.path.join(path_A, fname)
                    src_GT = os.path.join(path_GT, fname)
                    dst_A = os.path.join(A_dir, fname)
                    dst_GT = os.path.join(GT_dir, fname)
                    shutil.copy(src_A, dst_A)
                    shutil.copy(src_GT, dst_GT)
    # 统计
    print("\n=== 数据集划分完成 ===")
    for s in ["train", "val", "test"]:
        cnt = len(os.listdir(os.path.join(png_root, f"{s}A")))
        print(f"{s}A: {cnt} 张")


if __name__ == "__main__":  
    RAW_DCM_ROOT = r"E:\liuying\data\BraTS_dicom"  # 原始DICOM根目录
    PNG_OUTPUT_ROOT = r"E:\liuying\data\BraTS20"     # PNG输出根目录
    # 第一步：DICOM转为PNG
    # dicom_to_png_raw(
    #     dcm_root=RAW_DCM_ROOT,
    #     output_root=PNG_OUTPUT_ROOT,
    #     sampling_step=SAMPLING_STEP,
    #     select_patient_num=20
    # )
    # 第二步：过滤全黑切片
    # filter_black_slices(PNG_OUTPUT_ROOT, 8, True)
    # 第三步：按患者划分 train/val/test
    split_dataset_by_patient(
        png_root=PNG_OUTPUT_ROOT,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=6
    )
