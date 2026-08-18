import os
import pandas as pd
import matplotlib.pyplot as plt

def plot(df_path, png_save_path):
    df = pd.read_csv(df_path)
    epoch = df["epoch"]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --------子图1 Generator Loss--------
    ax1 = axes[0,0]
    ax1.plot(epoch, df["train_G_loss"], label="Train G Loss", color="#1f77b4")
    ax1.plot(epoch, df["Val_G_Loss"], label="Val G Loss", color="#ff7f0e")
    ax1.set_title("Generator Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --------子图2 Discriminator Loss--------
    ax2 = axes[0,1]
    ax2.plot(epoch, df["train_D_loss"], label="Train D Loss", color="#2ca02c")
    ax2.plot(epoch, df["Val_D_Loss"], label="Val D Loss", color="#d62728")
    ax2.set_title("Discriminator Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --------子图3 PSNR--------
    ax3 = axes[1,0]
    ax3.plot(epoch, df["val_psnr"], color="#9467bd", label="Val PSNR")
    ax3.set_title("Validation PSNR")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # --------子图4 SSIM & Dice--------
    ax4 = axes[1,1]
    ax4.plot(epoch, df["val_ssim"], label="Val SSIM", color="#8c564b")
    ax4.plot(epoch, df["val_dice"], label="Val Dice", color="#e377c2")
    ax4.set_title("Validation SSIM & Dice")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Metric Value")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ =="__main__":
    # 读取日志
    df_path = r"E:\mycode\cyclegan\log\exp5\train_log.csv"
    save_root = os.path.dirname(df_path)
    png_save_path = os.path.join(save_root, "train_log.png")
    # print(OUTPUT_DIR)
    plot(df_path, png_save_path)