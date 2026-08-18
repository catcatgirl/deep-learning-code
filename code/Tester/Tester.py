# Tester/Tester.py
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from Model.CycleGAN import DeepGAD


class Tester:
    def __init__(self,
                 ckpt_path: str,
                 input_dir: str,
                 output_dir: str,
                 img_size: int = 256,
                 device: str = None):
        """
        :param ckpt_path: 权重文件路径
        :param input_dir: 输入png切片文件夹(testA)
        :param output_dir: 输出预测结果文件夹
        :param img_size: 图像尺寸
        :param device: cuda / cpu，不传自动判断
        """
        self.ckpt_path = ckpt_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """加载权重"""
        self.model = DeepGAD().to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.eval()
        print("[Tester] 模型加载完成")

    def infer_single_slice(self, img_ncct_path: str):
        """单张切片推理"""
        img_ncct = cv2.imread(img_ncct_path, cv2.IMREAD_GRAYSCALE)
        if img_ncct is None:
            print(f"[WARN] 读取图片失败: {img_ncct_path}")
            return None

        resize = T.Resize((self.img_size, self.img_size))
        img_ncct = torch.from_numpy(img_ncct / 255.0).float().unsqueeze(0).unsqueeze(0)
        input_2ch = resize(img_ncct).to(self.device) * 2 - 1

        with torch.no_grad():
            pred = self.model.G_A(input_2ch)

        pred_np = ((pred + 1) / 2).squeeze().cpu().numpy() * 255
        pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
        return pred_np

    def run(self):
        """执行全套推理，对外入口，和trainer.run_pipe()对齐"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_model()

        slice_list = sorted([f for f in os.listdir(self.input_dir) if f.endswith(".png")])
        print(f"[Tester] 共 {len(slice_list)} 张切片，开始推理")

        for idx, slice_name in enumerate(slice_list):
            ncct_path = os.path.join(self.input_dir, slice_name)
            pred_img = self.infer_single_slice(ncct_path)
            if pred_img is not None:
                cv2.imwrite(os.path.join(self.output_dir, slice_name), pred_img)

            if (idx + 1) % 20 == 0:
                print(f"[Tester] 已完成 {idx+1}/{len(slice_list)}")

        print(f"[Tester] 推理全部完成，输出路径：{self.output_dir}")