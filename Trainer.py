import os
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Log import Log
from skimage import filters
from ModelDeepgad import DeepGAD
from DatasetPng import CCTADataset
from CombinedGANEarlyStopping import CombinedGANEarlyStopping
from Metric import calculate_psnr_ssim, calculate_vessel_dice


class Trainer:
    def __init__(self, train_batch_size:int, valid_batch_size:int, train_A_path:object, train_GT_path:object, valid_A_path:object, valid_GT_path:object, 
                 lr:float, epochs:int = 50, log:object = None, earlystopping:object = None, model:object = None, metrics:object = None, 
                 save_path_dir:object = None) -> None:
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_A_path = train_A_path
        self.train_GT_path = train_GT_path
        self.valid_A_path = valid_A_path
        self.valid_GT_path = valid_GT_path
        self.log = log
        self.lr = lr
        self.epochs = epochs
        self.save_path_dir = save_path_dir
        self._early_stopping = earlystopping
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        # 损失权重（冠脉优化版）
        self.LAMBDA_L1 = 100
        self.LAMBDA_CYCLE = 10
        self.LAMBDA_VESSEL = 25
    
    def loss_func(self,name):
        if name == 'MSEloss':
            loss = nn.MSELoss()
        elif name == 'L1loss':
            loss = nn.L1Loss()
        return loss

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def activate_log(self):
        print('Activate log...')
        self.log.open()
        # self.save_path = self.log.get_save_path()
        self.log.activate()
        self.log.log('\n\nTrain Model\n\n')
        
    def load_configuration(self):
        self.parallel = False   # ←修复：暂时关闭DataParallel，避免权重加载坑，后续需要再开启
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log.log('======= Parameters =======')
        self.log.log(f'Epochs: {self.epochs}')
        self.log.log(f'Train Batch Size: {self.train_batch_size}')
        self.log.log(f'Valid Batch Size: {self.valid_batch_size}')
        self.log.log(f'Learning Rate: {self.lr}')
        self.log.log(f'Parallel: {self.parallel}')
        self.log.log(f'Device: {self.device}')
        self.log.log('==========================\n')
        print('Load Configuration Finished...')
        
    def load_data(self):
        train_data = CCTADataset(self.train_A_path, self.train_GT_path, is_train=True, img_size=256)
        valid_data = CCTADataset(self.valid_A_path, self.valid_GT_path, is_train=False, img_size=256)
        # 根据是否存在GPU自动开启pin_memory，消除警告
        use_pin = torch.cuda.is_available()
        self.train_loader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True, num_workers=0, pin_memory=use_pin)
        self.valid_loader = DataLoader(valid_data, batch_size=self.valid_batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)
        print('Load Data Finished...')

    def load_model(self):
        self.model = DeepGAD().to(self.device)  
        self.log.log('======== Model ========')
        self.log.log(self.model)
        self.log.log('=======================\n')
        # ←修复：删除了初始化时错误的保存模型代码
        print('Load Model Finished...')

    def vessel_loss(self, pred, gt):
        """
        【重要提醒】当前实现存在梯度断裂问题！
        skimage+numpy会脱离计算图，暂时不要启用！
        如需血管损失，后续需要重写Pytorch可微分Frangi
        """
        batch = pred.shape[0]
        total_loss = 0.0
        for i in range(batch):
            pred_np = pred[i, 0].detach().cpu().numpy()
            gt_np = gt[i, 0].detach().cpu().numpy()
            pred_vessel = filters.frangi(pred_np, sigmas=[0.5, 1, 1.5])
            gt_vessel = filters.frangi(gt_np, sigmas=[0.5, 1, 1.5])
            total_loss += np.mean(np.abs(pred_vessel - gt_vessel))
        return torch.tensor(total_loss / batch, requires_grad=True).to(self.device)
    
    def train_one_epoch(self, opt_G, opt_D):
        self.model.train()
        total_G_loss = 0.0
        total_D_loss = 0.0
        for input_2ch, gt_full in self.train_loader:
            input_2ch = input_2ch.to(self.device)
            gt_full = gt_full.to(self.device)
            batch = input_2ch.shape[0]
            real_label = torch.ones((batch, 1, 30, 30)).to(self.device) * 0.9
            fake_label = torch.zeros((batch, 1, 30, 30)).to(self.device)
            opt_G.zero_grad()
            fake_full = self.model.G_A(input_2ch)
            pred_fake = self.model.D_A(fake_full)

            loss_adv_G = self.criterion_gan(pred_fake, real_label)
            loss_l1_G = self.criterion_l1(fake_full, gt_full) * self.LAMBDA_L1
            rec_2ch = self.model.G_B(fake_full)
            loss_cycle = self.criterion_l1(rec_2ch, input_2ch) * self.LAMBDA_CYCLE

            # ←修复：暂时注释血管损失，防止梯度断裂！！
            # loss_vessel = self.vessel_loss(fake_full, gt_full) * self.LAMBDA_VESSEL
            loss_vessel = 0.0

            loss_G = loss_adv_G + loss_cycle + loss_l1_G + loss_vessel
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt_G.step()

            opt_D.zero_grad()
            pred_real = self.model.D_A(gt_full)
            loss_D_real = self.criterion_gan(pred_real, real_label)
            pred_fake = self.model.D_A(fake_full.detach())
            loss_D_fake = self.criterion_gan(pred_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            opt_D.step()

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
        avg_G_loss = total_G_loss / len(self.train_loader)
        avg_D_loss = total_D_loss / len(self.train_loader)
        return avg_G_loss, avg_D_loss

    def validate(self):
        self.model.eval()
        total_G_loss = 0.0
        total_D_loss = 0.0
        psnr_all, ssim_all, dice_all = [], [], []
        with torch.no_grad():
            for input_2ch, gt_full in self.valid_loader:
                input_2ch = input_2ch.to(self.device)
                gt_full = gt_full.to(self.device)
                pred_full = self.model.G_A(input_2ch)                
                loss_l1 = self.criterion_l1(pred_full, gt_full) * self.LAMBDA_L1                
                rec_2ch = self.model.G_B(pred_full)
                loss_cycle = self.criterion_l1(rec_2ch, input_2ch) * self.LAMBDA_CYCLE                
                loss_G = loss_l1 + loss_cycle

                batch = input_2ch.shape[0]
                real_label = torch.ones((batch, 1, 30, 30)).to(self.device) * 0.9
                fake_label = torch.zeros((batch, 1, 30, 30)).to(self.device)                
                pred_real = self.model.D_A(gt_full)
                loss_D_real = self.criterion_gan(pred_real, real_label)                
                pred_fake = self.model.D_A(pred_full)
                loss_D_fake = self.criterion_gan(pred_fake, fake_label)                
                loss_D = (loss_D_real + loss_D_fake) / 2                
                total_G_loss += loss_G.item()
                total_D_loss += loss_D.item()                

                gt_np = ((gt_full + 1) / 2).squeeze().cpu().numpy()
                pred_np = ((pred_full + 1) / 2).squeeze().cpu().numpy()
                p, s = calculate_psnr_ssim(gt_np, pred_np)
                d = calculate_vessel_dice(gt_np, pred_np)
                psnr_all.append(p)
                ssim_all.append(s)
                dice_all.append(d)
        avg_G_loss = total_G_loss / len(self.valid_loader)
        avg_D_loss = total_D_loss / len(self.valid_loader)
        return np.mean(psnr_all), np.mean(ssim_all), np.mean(dice_all), avg_G_loss, avg_D_loss
    
    # ←修复：增加静态方法装饰器
    @staticmethod
    def caltime(s):
        h = int(s / 3600)
        m = int((s - h * 3600) / 60)
        s = int(s - h * 3600 - m * 60)
        return f'{h}:{m}:{s}'
    
    def train_model(self):
        print('Train Action...')
        RESUME_CKPT = None
        BETAS = (0.5, 0.999)
        opt_G = optim.Adam(
            list(self.model.G_A.parameters()) + list(self.model.G_B.parameters()),
            lr=self.lr, betas=BETAS
        )
        opt_D = optim.Adam(
            list(self.model.D_A.parameters()) + list(self.model.D_B.parameters()),
            lr=self.lr, betas=BETAS
        )
        scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=30, gamma=0.5)
        scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=30, gamma=0.5)

        start_epoch = 1
        best_dice = 0.0
        log_history = []
        if RESUME_CKPT and os.path.exists(RESUME_CKPT):
            checkpoint = torch.load(RESUME_CKPT, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            opt_G.load_state_dict(checkpoint["opt_G"])
            opt_D.load_state_dict(checkpoint["opt_D"])
            start_epoch = checkpoint["epoch"] + 1
            best_dice = checkpoint["best_dice"]
            print(f"加载断点权重，从第{start_epoch}轮继续训练")
        actual_log_dir = self.log.save_path
        SAVE_CKPT_DIR = os.path.join(actual_log_dir, "checkpoints")
        os.makedirs(SAVE_CKPT_DIR, exist_ok=True)
        self.early_stopping = CombinedGANEarlyStopping(save_path=SAVE_CKPT_DIR, patience=20, verbose=True)
        all_start_time = time.time()

        for epoch in range(start_epoch, self.epochs + 1):
            start_time = time.time()
            train_g_loss, train_d_loss = self.train_one_epoch(opt_G, opt_D)
            val_psnr, val_ssim, val_dice, val_g_loss, val_d_loss = self.validate()
            end_time = time.time()
            scheduler_G.step()
            scheduler_D.step()

            # ←修复：{self.epochs+1} 改成 {self.epochs}
            self.log.log(f"Epoch:  [{epoch:03d}/{self.epochs}] | "
                    f"Train_G_Loss:{train_g_loss:.4f} | Train_D_Loss:{train_d_loss:.4f} |  "
                    f"PSNR:{val_psnr:.4f} | SSIM:{val_ssim:.4f} | Vessel_Dice:{val_dice:.4f} | "
                    f"Valid_G_Loss:{val_g_loss:.4f} | Valid_D_Loss:{val_d_loss:.4f} | "
                    f"Use_time: {self.caltime(end_time - start_time)}")
            log_history.append({
                "epoch": epoch, "train_G_loss": train_g_loss, "train_D_loss": train_d_loss,
                "val_psnr": val_psnr, "val_ssim": val_ssim, "val_dice": val_dice,
                "Val_G_Loss":val_g_loss, "Val_D_Loss":val_d_loss
            })
        
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), os.path.join(SAVE_CKPT_DIR, "best_model_by_dice.pth"))
                print(f"★ 最优模型已更新（Dice），Dice={best_dice:.4f}")
            
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "best_dice": best_dice
            }, os.path.join(SAVE_CKPT_DIR, "latest.pth"))
            
            self.early_stopping(val_g_loss, val_d_loss, val_dice, self.model, epoch)
            if self.early_stopping.early_stop:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break   # ←修复：触发早停直接跳出循环
        all_end_time = time.time()        
        self.log.log(f'Train_use_time: {self.caltime(all_end_time - all_start_time)}')    
        self.log.deactivate()
    
        pd.DataFrame(log_history).to_csv(os.path.join(actual_log_dir, "train_log.csv"), index=False)
        
        print("训练全部完成！")
        print(f"最优Dice: {best_dice:.4f}")
        print(f"最优模型保存在: {os.path.join(SAVE_CKPT_DIR, 'best_model_by_dice.pth')}")
        
    def save_losses_to_csv(self, losses):
        # 该函数存在未定义变量，暂时保留不调用
        pass

    def deactivate_log(self):
        self.log.deactivate()
        print('Deactivate log...')

    def run_pipe(self):
        self.set_seed(seed=6)   # ←修复：固定随机种子，保证实验可复现
        self.activate_log()
        self.load_configuration()
        self.load_data()
        self.load_model()
        self.train_model()
        self.deactivate_log()

if __name__ == "__main__":
    # 数据路径
    TRAIN_A = r"E:\data\png\trainA"
    TRAIN_GT = r"E:\data\png\trainGT"
    VAL_A = r"E:\data\png\valA"
    VAL_GT = r"E:\data\png\valGT"
    LOG_DIR = r"E:\AI_anke\MyCode\logs"    
    os.makedirs(LOG_DIR, exist_ok=True)
    log = Log(LOG_DIR)
    # early_stopping = CombinedGANEarlyStopping(save_path=LOG_DIR, patience=20, verbose=True)
    
    tm = Trainer(
        train_batch_size=1,
        valid_batch_size=1,
        train_A_path=TRAIN_A,
        train_GT_path=TRAIN_GT,
        valid_A_path=VAL_A,
        valid_GT_path=VAL_GT, 
        lr=0.0002,
        epochs=1,
        log=log,
        save_path_dir=LOG_DIR
    )
    tm.run_pipe()
    