import torch
import os
import cv2
import numpy as np
import torchvision.transforms as T
from ModelDeepgad import DeepGAD

# ===================== 推理参数 =====================
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = r"E:\liuying\MyCode\deep-learning-code\logs\exp8\checkpoints\best_model_by_dice.pth"
# 患者输入文件夹
INPUT_NCCT_DIR = r"E:\liuying\data\test\BraTS_png\testA"
exp_root = os.path.dirname(os.path.dirname(CKPT_PATH))
OUTPUT_DIR = os.path.join(exp_root, "test")
# =====================================================

def load_model():
    """加载训练好的模型"""
    model = DeepGAD().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载完成")
    return model

def infer_single_slice(model, img_ncct_path):
    """推理单张切片"""
    # 读取图片
    img_ncct = cv2.imread(img_ncct_path, cv2.IMREAD_GRAYSCALE)
    # img_low = cv2.imread(img_low_path, cv2.IMREAD_GRAYSCALE)

    # 预处理
    resize = T.Resize((IMG_SIZE, IMG_SIZE))
    img_ncct = torch.from_numpy(img_ncct / 255.0).float().unsqueeze(0).unsqueeze(0)
    # img_low = torch.from_numpy(img_low / 255.0).float().unsqueeze(0).unsqueeze(0)
    # input_2ch = torch.cat([img_ncct, img_low], dim=1)
    input_2ch = resize(img_ncct).to(DEVICE) * 2 - 1

    # 推理
    with torch.no_grad():
        pred = model.G_A(input_2ch)
    # 后处理：[-1,1]转回0-255灰度
    pred_np = ((pred + 1) / 2).squeeze().cpu().numpy() * 255
    pred_np = np.clip(pred_np, 0, 255).astype(np.uint8)
    return pred_np

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()

    # 遍历所有切片
    slice_list = sorted([f for f in os.listdir(INPUT_NCCT_DIR) if f.endswith(".png")])
    print(f"共{len(slice_list)}张切片，开始推理...")

    for idx, slice_name in enumerate(slice_list):
        ncct_path = os.path.join(INPUT_NCCT_DIR, slice_name)
        # low_path = os.path.join(INPUT_LOWDOSE_DIR, slice_name)
        pred_img = infer_single_slice(model, ncct_path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, slice_name), pred_img)
        if (idx+1) % 20 == 0:
            print(f"已完成 {idx+1}/{len(slice_list)} 张")

    print(f"全部推理完成！合成造影图保存在：{OUTPUT_DIR}")