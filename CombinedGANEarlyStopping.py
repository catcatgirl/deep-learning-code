import os
import torch
import numpy as np
import torch.nn as nn

class CombinedGANEarlyStopping:
    def __init__(self, save_path, patience=10, verbose=True):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_dice = 0
        self.best_G_loss = np.inf
        self.best_D_loss = np.inf
        self.early_stop = False
        os.makedirs(save_path, exist_ok=True)
        
    def __call__(self, val_G_loss, val_D_loss, val_dice, model, epoch):
        # G损失改善
        g_improved = val_G_loss < self.best_G_loss * 0.99
        # D损失在合理范围内
        d_reasonable = 0.3 < val_D_loss < 3.0
        # Dice改善
        dice_improved = val_dice > self.best_dice + 0.002
        # 判断是否保存模型：G改善且Dice改善
        if g_improved and dice_improved:
            self.best_G_loss = val_G_loss
            self.best_D_loss = val_D_loss
            self.best_dice = val_dice
            self.counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'G_loss': val_G_loss,
                'D_loss': val_D_loss,
                'dice': val_dice
            }, os.path.join(self.save_path, 'best_model_by_loss_and_dice.pth'))
            
            if self.verbose:
                print(f'✓ Model improved: G={val_G_loss:.4f}, D={val_D_loss:.4f}, Dice={val_dice:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'⏳ No improvement: {self.counter}/{self.patience} '
                      f'(G={val_G_loss:.4f}, D={val_D_loss:.4f}, Dice={val_dice:.4f})')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'🛑 Early stopping triggered at epoch {epoch}')
                    
        def load_best_model(self, model):
            
            """加载最优模型"""
            path = os.path.join(self.save_path, 'best_model.pth')
            if os.path.exists(path):
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch with Dice={checkpoint['dice']:.4f}")
            return model
