import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset_by_patient(png_root, val_ratio=0.2, test_ratio=0.2, random_state=6):
    """
    从 A / GT 中按患者ID拆分数据集
    """
    path_A = os.path.join(png_root, "A")
    path_GT = os.path.join(png_root, "GT")
    # 从文件名提取患者ID
    img_files = [f for f in os.listdir(path_A) if f.endswith(".png")]
    patient_ids = sorted(list({f.split("_")[0] for f in img_files}))
    print(f"共 {len(patient_ids)} 名患者，开始划分数据集")
    # 划分患者
    train_val_pids, test_pids = train_test_split(patient_ids, test_size=test_ratio, random_state=random_state)
    train_pids, val_pids = train_test_split(train_val_pids, test_size=val_ratio/(1-test_ratio), random_state=random_state)
    print(f"训练患者: {len(train_pids)}")
    print(f"验证患者: {len(val_pids)}")
    print(f"测试患者: {len(test_pids)}")
    # 创建目录
    subsets = [
        ("train", train_pids),
        ("val", val_pids),
        ("test", test_pids)
    ]

    for subset_name, pids in subsets:
        A_dir = os.path.join(png_root, f"{subset_name}A")
        GT_dir = os.path.join(png_root, f"{subset_name}GT")
        os.makedirs(A_dir, exist_ok=True)
        os.makedirs(GT_dir, exist_ok=True)
        for pid in pids:
            for fname in img_files:
                if fname.startswith(pid + "_"):
                    src_A = os.path.join(path_A, fname)
                    src_GT = os.path.join(path_GT, fname)
                    dst_A = os.path.join(A_dir, fname)
                    dst_GT = os.path.join(GT_dir, fname)
                    shutil.copy(src_A, dst_A)
                    shutil.copy(src_GT, dst_GT)
    # 统计
    print("\n=== 数据集划分完成 ===")
    for s in ["train", "val", "test"]:
        cnt = len(os.listdir(os.path.join(png_root, f"{s}A")))
        print(f"{s}A: {cnt} 张")

if __name__ == "__main__":  
    #数据划分比例
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    SAMPLING_STEP=0
    PNG_ROOT = r"E:\data\head_neck_png2"
    # 第三步：按患者划分 train/val/test
    split_dataset_by_patient(
        png_root=PNG_ROOT,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=6
    )

    '''
    训练患者: 2
    验证患者: 1
    测试患者: 1

    trainA: 143 张
    valA: 71 张
    testA: 68 张
    '''