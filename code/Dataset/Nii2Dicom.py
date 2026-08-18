import os
import uuid
import numpy as np
import nibabel as nib
from pydicom import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian

# =================配置路径=================
nii_path = r"E:\data\BraTS-GLI-00000-000-seg.nii\BraTS-GLI-00000-000-seg.nii"
dicom_out_dir = r"E:\AI_anke\MyCode\data\Nii2Dicom\seg"
os.makedirs(dicom_out_dir, exist_ok=True)

# 读取NIfTI
nii_img = nib.load(nii_path)
vol = nii_img.get_fdata()
affine = nii_img.affine
spacing = nii_img.header.get_zooms()
H, W, D = vol.shape

# ==========关键：统一UID，所有切片共用一套Study/Series UID==========
study_uid = str(uuid.uuid4())
series_uid = str(uuid.uuid4())

# 循环生成每一层DICOM
for z in range(D):
    arr = vol[:, :, z]
    # 创建DICOM数据集
    ds = FileDataset(None, {}, preamble=b"\x00"*128)
    ds.file_meta = Dataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    # 患者信息
    ds.PatientName = "BraTS Patient"
    ds.PatientID = "BraTS00000"

    # 检查、序列信息【核心】
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = str(uuid.uuid4())  # 每张必须唯一
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
    save_path = os.path.join(dicom_out_dir, f"slice_{z:03d}.dcm")
    ds.save_as(save_path)

print("转换完成！全部切片属于同一个DICOM序列")