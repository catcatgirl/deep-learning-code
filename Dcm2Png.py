import os
import cv2
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split

# ===================== 冠脉CTA专用参数 =====================
# HU_MIN = -500    # 低于空气、肺组织的HU值截断
# HU_MAX = 1000    # 高于钙化、支架的HU值截断

HU_MIN = -200
HU_MAX = 800
VAL_RATIO = 0.2  # 验证集占比
RAW_DCM_ROOT = r"E:\data\test\test"  # 原始DICOM根目录
PNG_OUTPUT_ROOT = r"E:\data\test\png2"      # PNG输出根目录
# =========================================================

def hu_to_gray(dcm_file_path):
    """
    CT图像
    单张DICOM转灰度PNG：HU值 -> 0-255灰度
    :param dcm_file_path: 单张.dcm文件路径
    :return: 0-255 uint8灰度图
    """
    dcm = pydicom.dcmread(dcm_file_path)
    # # 原始像素值转HU亨氏单位
    # hu_array = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # # HU截断，去除无关极端值
    # hu_clipped = np.clip(hu_array, HU_MIN, HU_MAX)
    # 安全获取Rescale参数
    slope = getattr(dcm, 'RescaleSlope', 1.0)
    intercept = getattr(dcm, 'RescaleIntercept', 0.0)
    
    # 获取像素数据
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    # 转换为HU值
    hu_array = pixel_array * slope + intercept
    
    # HU截断
    hu_clipped = np.clip(hu_array, HU_MIN, HU_MAX)
    
    # 线性映射到0-255灰度范围
    gray_255 = ((hu_clipped - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
    return gray_255

def get_all_dcm(folder):
    file_list = []
    for root, _, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            # 跳过文件夹，只判断文件
            if os.path.isfile(full_path):
                try:
                    # 尝试读取，能打开就是DICOM
                    pydicom.dcmread(full_path, stop_before_pixels=True)
                    file_list.append(full_path)
                    # print(f"识别到DICOM：{full_path}")
                except:
                    # 不是DICOM直接跳过
                    continue
    return sorted(file_list)

def mri_to_gray(dcm_file_path):
    """
    MRI单张DICOM转灰度PNG：自适应百分位截断，保留全信息
    完全对齐DeepGAD原文的直方图强度归一化逻辑
    """
    # 百分位截断：去掉上下1%的极端噪声点，保留99%的有效组织信息
    # 要100%保留所有信号就改成0和100，会受少量噪声影响
    PERCENTILE_LOW = 1
    PERCENTILE_HIGH = 99

    dcm = pydicom.dcmread(dcm_file_path)
    # MRI直接用原始像素值，不需要转HU（HU是CT专属）
    pixel_array = dcm.pixel_array.astype(np.float32)

    # 自适应百分位截断：去掉极端噪声伪影，保留有效组织的全部动态范围
    low = np.percentile(pixel_array, PERCENTILE_LOW)
    high = np.percentile(pixel_array, PERCENTILE_HIGH)
    # 截断到有效范围
    img_clipped = np.clip(pixel_array, low, high)
    # 线性映射到0-255灰度，充分利用8位灰度的动态范围
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
                    # 优先用SliceLocation，没有就用InstanceNumber
                    slice_pos = getattr(dcm, 'SliceLocation', 0.0)
                    instance_num = getattr(dcm, 'InstanceNumber', 0)
                    dcm_list.append((slice_pos, instance_num, full_path))
                except:
                    continue
    # 按Z轴位置从小到大排序，保证层序正确
    dcm_list.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in dcm_list]



def process_CT_single_patient(patient_dir, patient_id, is_train,SAMPLING_STEP):
    """
    CT图像
    处理单个患者的三组DICOM，生成对应PNG
    """
    ncct_dir = os.path.join(patient_dir, "t1n")
    full_dir = os.path.join(patient_dir, "t1c")

    # 递归深层遍历所有dcm，兼容子文件夹
    ncct_files = get_all_dcm(ncct_dir)
    # low_files = get_all_dcm(lowdose_dir)
    full_files = get_all_dcm(full_dir)
    print(f"【{patient_id}】ncct:{len(ncct_files)}  full:{len(full_files)}")

    # 校验三组切片数量一致
    assert len(ncct_files) == len(full_files), \
        f"患者{patient_id}三组序列切片数量不一致，请检查数据！"

    # 区分训练/验证文件夹
    if is_train:
        target_A = os.path.join(PNG_OUTPUT_ROOT, "trainA")
        # target_B = os.path.join(PNG_OUTPUT_ROOT, "trainB")
        target_GT = os.path.join(PNG_OUTPUT_ROOT, "trainGT")
    else:
        target_A = os.path.join(PNG_OUTPUT_ROOT, "valA")
        # target_B = os.path.join(PNG_OUTPUT_ROOT, "valB")
        target_GT = os.path.join(PNG_OUTPUT_ROOT, "valGT")

    # 创建文件夹
    for folder in [target_A, target_GT]:
        os.makedirs(folder, exist_ok=True)
        
        

    for idx, (ncct_f, full_f) in enumerate(zip(ncct_files, full_files)):
        
        # =========核心改动：每5张取1张==========
        if idx % SAMPLING_STEP != 0:
            continue
        
        # 生成统一文件名：患者ID_切片号.png
        img_name = f"{patient_id}_{idx:04d}.png"

        # 转灰度图
        img_ncct = hu_to_gray(ncct_f)
        # img_low = hu_to_gray(low_f)
        img_full = hu_to_gray(full_f)

        # 保存PNG
        cv2.imwrite(os.path.join(target_A, img_name), img_ncct)
        # cv2.imwrite(os.path.join(target_B, img_name), img_low)
        cv2.imwrite(os.path.join(target_GT, img_name), img_full)

    print(f"患者 {patient_id} 处理完成，共{len(ncct_files)}张切片")


def process_MRI_single_patient(patient_dir, patient_id, is_train, is_test, SAMPLING_STEP):
    """MRI：处理单个患者的平扫T1 + 增强T1两组MRI"""
    ncct_dir = os.path.join(patient_dir, "t1n")  # 平扫T1（对应原文的pre-contrast）
    full_dir = os.path.join(patient_dir, "t1c")  # 增强T1（对应原文的full-dose）

    # 按空间位置排序
    ncct_files = get_sorted_dcm(ncct_dir)
    full_files = get_sorted_dcm(full_dir)
    print(f"【{patient_id}】平扫T1:{len(ncct_files)}张 增强T1:{len(full_files)}张")

    # 校验切片数量一致
    assert len(ncct_files) == len(full_files), \
        f"患者{patient_id}平扫和增强切片数量不一致，请检查数据！"

    # 区分训练/验证文件夹
    if is_train:
        target_A = os.path.join(PNG_OUTPUT_ROOT, "trainA")
        target_GT = os.path.join(PNG_OUTPUT_ROOT, "trainGT")
    elif is_test:
        target_A = os.path.join(PNG_OUTPUT_ROOT, "testA")
        target_GT = os.path.join(PNG_OUTPUT_ROOT, "testGT")
    else:
        target_A = os.path.join(PNG_OUTPUT_ROOT, "valA")
        target_GT = os.path.join(PNG_OUTPUT_ROOT, "valGT")

    for folder in [target_A, target_GT]:
        os.makedirs(folder, exist_ok=True)

    for idx, (ncct_f, full_f) in enumerate(zip(ncct_files, full_files)):
        # =========核心改动：每5张取1张==========
        if idx % SAMPLING_STEP != 0:
            continue
        img_name = f"{patient_id}_{idx:04d}.png"
        img_ncct = mri_to_gray(ncct_f)
        img_full = mri_to_gray(full_f)
        cv2.imwrite(os.path.join(target_A, img_name), img_ncct)
        cv2.imwrite(os.path.join(target_GT, img_name), img_full)

    print(f"患者 {patient_id} 处理完成，共{len(ncct_files)}张切片")


if __name__ == "__main__":
    # 遍历所有患者文件夹
    patient_list = sorted([
        p for p in os.listdir(RAW_DCM_ROOT)
        if os.path.isdir(os.path.join(RAW_DCM_ROOT, p))
    ])
    # print(patient_list)
    print(f"共找到 {len(patient_list)} 名患者数据，开始转换...")
    # 所有患者 → 测试集
    for pid in patient_list:
        patient_path = os.path.join(RAW_DCM_ROOT, pid)
        # is_train=False, is_test=True
        process_MRI_single_patient(patient_path, pid, is_train=False, is_test=True, SAMPLING_STEP=5)

    print("全部DICOM转PNG完成！")
    testA_path = os.path.join(PNG_OUTPUT_ROOT, "testA")
    testGT_path = os.path.join(PNG_OUTPUT_ROOT, "testGT")
    print(f"测试集输入切片数量：{len(os.listdir(testA_path))} 张")
    print(f"测试集真值切片数量：{len(os.listdir(testGT_path))} 张")

    # # 按患者划分训练/验证（医学数据标准做法，避免数据泄露）
    # train_patients, val_patients = train_test_split(patient_list, test_size=VAL_RATIO, random_state=42)
    # print(f"训练患者数：{len(train_patients)}，验证患者数：{len(val_patients)}")

    # # 处理训练集患者
    # for pid in train_patients:
    #     patient_path = os.path.join(RAW_DCM_ROOT, pid)
    #     process_MRI_single_patient(patient_path, pid, is_train=True, is_test=False, SAMPLING_STEP=5)
    # # 处理验证集患者
    # for pid in val_patients:
    #     patient_path = os.path.join(RAW_DCM_ROOT, pid)
    #     process_MRI_single_patient(patient_path, pid, is_train=False, is_test=False, SAMPLING_STEP=5)

    # print("全部DICOM转PNG完成！")
    # # 修正路径，使用全局PNG_OUTPUT_ROOT
    # trainA_path = os.path.join(PNG_OUTPUT_ROOT, "trainA")
    # valA_path = os.path.join(PNG_OUTPUT_ROOT, "valA")
    # print(f"训练集切片数量：{len(os.listdir(trainA_path))} 张")
    # print(f"验证集切片数量：{len(os.listdir(valA_path))} 张")