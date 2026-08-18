import os
import cv2
import shutil
import numpy as np

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

if __name__ == "__main__":  
    PNG_ROOT = r"E:\data\head_neck_png2"     # PNG根目录
    filter_black_slices(PNG_ROOT, 8, True)