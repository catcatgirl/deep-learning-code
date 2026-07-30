'''
out_root/
└── BraTS-GLI-00000-000/        # 一个病例一个总文件夹
    ├── t1c/                     # t1c序列的所有dcm
    └── t1n/                     # t1n序列的所有dcm
'''
#%%
import os
import uuid
import numpy as np
import nibabel as nib
from pydicom import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian
#%%
def FindBraTS(root_dir):
    """
    找到BraTS文件，并记录该数据集的nii文件路径，这里只记录t1c、t1n文件
    root_dir:BraTS开头的文件夹的根目录
    返回字典：{病例名称: {"t1c":路径, "t1n":路径}}
    """
    case_dict = {}
    for entry in os.scandir(root_dir):
        if entry.is_dir() and entry.name.startswith('BraTS'):
            case_name = entry.name
            case_folder = entry.path
            case_dict[case_name] = {}
            case_folder=entry.path
            for f in os.listdir(case_folder):
                fpath = os.path.join(case_folder, f)
                if f.endswith("-t1c.nii"):
                    case_dict[case_name]["t1c"] = fpath
                elif f.endswith("-t1n.nii"):
                    case_dict[case_name]["t1n"] = fpath
    return case_dict

def generate_valid_uid():
    """符合DICOM完整规范的UID，杜绝所有UI警告"""
    import random
    # 首数字1~9，避免0开头
    first = str(random.randint(1,9)) + "".join(str(random.randint(0,9)) for _ in range(31))
    return f"1.2.840.10008.{first}"

def Nii2Dicom(nii_path, save_dir):
    # 构造当前nii对应的输出子文件夹，避免文件覆盖
    os.makedirs(save_dir, exist_ok=True)
    
    nii_img = nib.load(nii_path)
    vol = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()
    H, W, D = vol.shape
    # 统一UID，所有切片共用一套Study/Series UID
    study_uid = generate_valid_uid()
    series_uid = generate_valid_uid()
    # 循环生成每一层DICOM
    for z in range(D):
        arr = vol[:, :, z]
        # 创建DICOM数据集
        ds = FileDataset(None, {}, preamble=b"\x00"*128)
        ds.file_meta = Dataset()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        # 患者信息
        ds.PatientName = "BraTS Patient"
        ds.PatientID = os.path.basename(os.path.dirname(nii_path))

        # 检查、序列信息【核心】
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_valid_uid()  # 每张必须唯一
        ds.InstanceNumber = z + 1

        # 图像基础信息
        ds.Rows = H
        ds.Columns = W
        ds.PixelSpacing = [spacing[0], spacing[1]]
        ds.SliceThickness = spacing[2]
        ds.SliceLocation = z * spacing[2]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0

        # 归一化像素，转uint16（MRI常用）
        arr_norm = arr - np.min(arr)
        if np.max(arr_norm) > 0:    
            arr_norm = arr_norm / np.max(arr_norm) * 4095
        arr_uint16 = arr_norm.astype(np.uint16)
        ds.PixelData = arr_uint16.tobytes()
        # 保存
        save_path = os.path.join(save_dir, f"slice_{z:03d}.dcm")
        ds.save_as(save_path)

#%%
if __name__ == "__main__":
    root_dir = r'E:\data'
    out_root = r'E:\data\AI data\BraTS'
    case_info  = FindBraTS(root_dir=root_dir)
    print(len(case_info ))
    for case_name, modal_dict in case_info.items():
        print(f"\n==== 正在处理病例：{case_name} ====")
        # t1c
        if "t1c" in modal_dict:
            nii_p = modal_dict["t1c"]
            target_dir = os.path.join(out_root, case_name, "t1c")
            print(f"转换 t1c -> {target_dir}")
            Nii2Dicom(nii_p, target_dir)
        # t1n
        if "t1n" in modal_dict:
            nii_p = modal_dict["t1n"]
            target_dir = os.path.join(out_root, case_name, "t1n")
            print(f"转换 t1n -> {target_dir}")
            Nii2Dicom(nii_p, target_dir)

    print("\n✅ 全部病例转换完成！")
# %%
