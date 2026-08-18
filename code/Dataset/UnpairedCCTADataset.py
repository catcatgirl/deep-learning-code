# Dataset.UnpairedCCTADataset.py
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class UnpairedCCTADataset(Dataset):
    def __init__(self, A_path, B_path, is_train=True, img_size=256):
        self.is_train = is_train
        self.img_size = img_size
        # A域：低剂量图像
        self.A_list = sorted([os.path.join(A_path, f) for f in os.listdir(A_path) if f.endswith(".png")])
        # B域：全剂量图像，和A不需要一一对应
        self.B_list = sorted([os.path.join(B_path, f) for f in os.listdir(B_path) if f.endswith(".png")])

    def __len__(self):
        # 取两者最大长度，保证每个epoch充分遍历
        return max(len(self.A_list), len(self.B_list))

    def __getitem__(self, index):
        idx_A = index % len(self.A_list)
        # 非配对关键：B随机采样，不跟A对齐
        idx_B = random.randint(0, len(self.B_list)-1)

        imgA = Image.open(self.A_list[idx_A]).convert("L")
        imgB = Image.open(self.B_list[idx_B]).convert("L")

        imgA = torch.from_numpy(np.array(imgA)).float() / 255.0
        imgB = torch.from_numpy(np.array(imgB)).float() / 255.0
        # [-1,1]归一化
        imgA = (imgA * 2) - 1
        imgB = (imgB * 2) - 1

        # 单通道扩展 [1,H,W]
        imgA = imgA.unsqueeze(0)
        imgB = imgB.unsqueeze(0)
        return imgA, imgB