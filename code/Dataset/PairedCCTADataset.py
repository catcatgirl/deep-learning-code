import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import numpy as np

class PairedCCTADataset(Dataset):
    """
    冠脉CCTA数据集：读取PNG，数据增强
    输入：平扫A 
    真值：全剂量GT 
    """
    def __init__(self, dir_A, dir_GT, dir_B=None, is_train=True, img_size=256):
        self.dir_A = dir_A  # trainA：NCCT平扫图
        self.dir_GT = dir_GT
        self.is_train = is_train  
        self.img_size = img_size
        # 所有图片文件名（严格排序保证对齐）
        self.img_names = sorted([f for f in os.listdir(dir_A) if f.endswith(".png")])
        # 训练集数据增强（冠脉专用，小范围变换避免解剖结构失真）
        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),          # 50%概率左右翻转， 水平翻转
                T.RandomVerticalFlip(p=0.2),            # 垂直翻转
                # 小角度旋转+轻微平移缩放，不破坏冠脉解剖结构
                T.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03)
                )
            ])
        else:
            self.transform = T.Resize((img_size, img_size))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # DataLoader 取数据时会循环调用该函数，输入索引返回 (输入图像, 真值图像)
        img_name = self.img_names[index]

        # 读取灰度图（0通道=灰度）
        img_A = cv2.imread(os.path.join(self.dir_A, img_name), cv2.IMREAD_GRAYSCALE)
        # img_B = cv2.imread(os.path.join(self.dir_B, img_name), cv2.IMREAD_GRAYSCALE)
        img_GT = cv2.imread(os.path.join(self.dir_GT, img_name), cv2.IMREAD_GRAYSCALE)

        # 转tensor，归一化到0-1
        img_A = torch.from_numpy(img_A / 255.0).float().unsqueeze(0)
        # img_B = torch.from_numpy(img_B / 255.0).float().unsqueeze(0)
        img_GT = torch.from_numpy(img_GT / 255.0).float().unsqueeze(0)

        # # 拼接双通道输入：[平扫, 低剂量] → shape [2, H, W]
        # input_2ch = torch.cat([img_A, img_B], dim=0)
        input_2ch = img_A

        # 数据增强（三张图一起增强，保证空间对齐）
        if self.is_train:
            combined = torch.cat([input_2ch, img_GT], dim=0)
            combined = self.transform(combined)
            input_2ch = combined[:1] # 前2通道=输入
            img_GT = combined[1:]    # 第3通道=真值
        else:
            input_2ch = self.transform(input_2ch)
            img_GT = self.transform(img_GT)

        # 映射到[-1, 1]，适配GAN的Tanh输出
        #  x ∈ [0,1] → x*2-1 ∈ [-1, 1]
        # 适配 GAN 生成器最后一层 Tanh 激活函数（Tanh 输出范围正好 - 1~1），是图像生成任务标准操作
        input_2ch = input_2ch * 2 - 1
        img_GT = img_GT * 2 - 1

        return input_2ch, img_GT
    
if __name__ == "__main__":
    PNG_ROOT = r'E:\data\png'
    # 训练集三个文件夹路径
    train_A = os.path.join(PNG_ROOT, "trainA")
    train_GT = os.path.join(PNG_ROOT, "trainGT")
    
    # 1. 路径校验
    for folder in [train_A, train_GT]:
        if not os.path.exists(folder):
            print(f"❌ 文件夹不存在：{folder}，请检查路径是否正确")
            exit()
    print("✅ 三个数据集文件夹路径校验通过")

    # 2. 实例化训练集
    try:
        train_dataset = PairedCCTADataset(train_A, train_GT, is_train=False, img_size=256)
        total_num = len(train_dataset)
        print(f"✅ 数据集加载成功，总样本数：{total_num} 张")
    except Exception as e:
        print(f"❌ 数据集实例化失败：{e}")
        exit()

    # 3. 测试第0张样本（单张图测试）
    try:
        input_2ch, gt_img = train_dataset[0]
        # 校验尺寸
        assert input_2ch.shape == (1, 256, 256), f"输入尺寸错误，预期[2,256,256]，实际{input_2ch.shape}"
        assert gt_img.shape == (1, 256, 256), f"真值尺寸错误，预期[1,256,256]，实际{gt_img.shape}"
        # 校验数值范围
        assert -1.0 <= input_2ch.min() and input_2ch.max() <= 1.0, "输入数值范围异常，不在[-1,1]之间"
        assert -1.0 <= gt_img.min() and gt_img.max() <= 1.0, "真值数值范围异常，不在[-1,1]之间"
        print(f"✅ 单样本加载通过：")
        print(f"   - 单通道输入尺寸：{input_2ch.shape}")
        print(f"   - 全剂量真值尺寸：{gt_img.shape}")
        print(f"   - 数值范围正常（适配Tanh的[-1,1]区间）")
        print(input_2ch)
    except Exception as e:
        print(f"❌ 单样本读取失败：{e}")


    # 4. 保存可视化结果，方便你肉眼核对是否正确
    save_test_dir = r"E:\AI_anke\MyCode\results\dataset_test_result"
    os.makedirs(save_test_dir, exist_ok=True)
    # 转回0-255灰度保存,逆归一化：[-1,1] → [0,255]，
    img_ncct = ((input_2ch[0] + 1) / 2 * 255).numpy().astype(np.uint8)
    img_full = ((gt_img[0] + 1) / 2 * 255).numpy().astype(np.uint8)
    
    cv2.imwrite(os.path.join(save_test_dir, "test_NCCT3.png"), img_ncct)
    cv2.imwrite(os.path.join(save_test_dir, "test_full3.png"), img_full)
    print(f"✅ 已保存测试可视化图到：{save_test_dir} 文件夹，可打开核对图像是否正确")

    # 5. 简单测试DataLoader批量加载（batch=1，验证训练时的加载流程）
    from torch.utils.data import DataLoader
    try:
        loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (1, 1, 256, 256), "批量输入尺寸错误"
        assert batch_y.shape == (1, 1, 256, 256), "批量真值尺寸错误"
        print(f"✅ DataLoader批量加载测试通过，训练时可正常读取")
    except Exception as e:
        print(f"❌ 批量加载失败：{e}")
        exit()

    print("="*60)
    print("🎉 数据集全部测试通过！可以直接开始训练了")
    print("💡 接下来修改train_png.py里的路径为你的E:\\data\\png即可启动训练")
    print("="*60)