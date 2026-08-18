import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from Utils.Log import Log
from Trainer.Trainer import Trainer
import gc
from Tester.Tester import Tester


# ====================== 配置 ======================
MODE = "train"          # 二选一： "train" 训练 / "test" 推理测试
# MODE = "test"
# --------训练参数--------
TRAIN_A = r"E:\data\head_neck_png\trainA"
TRAIN_GT = r"E:\data\head_neck_png\trainGT"
VAL_A = r"E:\data\head_neck_png\valA"
VAL_GT = r"E:\data\head_neck_png\valGT"
LOG_DIR = r"E:\mycode\deep-learning-code\logs"

# --------测试参数--------
CKPT_PATH = r"E:\mycode\deep-learning-code\logs\exp9\checkpoints\best_model_composite.pth"
INPUT_NCCT_DIR = r"E:\data\head_neck_png\test\A"
EXP_ROOT = os.path.dirname(os.path.dirname(CKPT_PATH))
OUTPUT_DIR = os.path.join(EXP_ROOT, "test_pre_result_onepicture")
IMG_SIZE = 256
# ====================================================


def run_test():
    tester = Tester(
        ckpt_path=CKPT_PATH,
        input_dir=INPUT_NCCT_DIR,
        output_dir=OUTPUT_DIR,
        img_size=IMG_SIZE
    )
    tester.run()


def run_train():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = Log(LOG_DIR)
    tm = Trainer(
        train_batch_size=1,
        valid_batch_size=1,
        train_A_path=TRAIN_A,
        train_GT_path=TRAIN_GT,
        valid_A_path=VAL_A,
        valid_GT_path=VAL_GT,
        lr=0.0002,
        epochs=100,
        log=log,
        save_path_dir=LOG_DIR
    )
    tm.run_pipe()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    if MODE == "train":
        run_train()
    elif MODE == "test":
        run_test()
    else:
        raise ValueError('MODE只能选 "train" 或者 "infer"')