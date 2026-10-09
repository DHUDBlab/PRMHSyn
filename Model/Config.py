# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:13:53 2023

@author: Mengjie Chen
"""

import argparse
import os
import torch

file = os.path.dirname(__file__) # current file

abs_path = os.path.abspath(os.path.join(file, os.pardir)) # project root

data_file = os.path.join(abs_path, "Data")


Remove_Repeated_Self_Loops = True


Use_Cell_Line_Feature = True



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse():
    p = argparse.ArgumentParser("PRMHSyn: PageRank-enhanced Relational Multimodal Hypergraph Learning for Drug Synergy Prediction", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset',type = str, default = 'ONEIL', help = 'the name of the dataset' ) # 'ONEIL', 'ALMANAC'
    p.add_argument('--model_name', type = str, default = 'PRMHSyn')
    p.add_argument('--threshold',type = int, default = 30,help = 'the threshold of positive samples') # 30, 10
    p.add_argument('--k_fold', type = int, default = 10, help = 'k-fold cross validation')
    p.add_argument('--cuda', type = str, default = '0', help = 'gpu id to use')
    p.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate')
    p.add_argument('--weight_decay', type = float, default = 1e-6, help = 'weight decay')
    p.add_argument('--epochs', type = int, default = 2000, help = 'number of epochs to train')
    p.add_argument('--seed', type = int, default = 2337, help = 'seed for randomness')

    # -----------------------------
    #  数据划分方式（与 Data_Process.py 对齐）
    # -----------------------------
    # 说明：
    # - cell_line: 原始“每个细胞系内部分割”（默认）
    # - cell_line_leaveout: 留出细胞系（test 细胞系未出现在 train/valid）
    # - drug_combo_leaveout: 留出药物组合
    # - drug_leaveout: 留出单个药物
    # - global: 全局随机划分
    p.add_argument(
        '--split_method',
        type=str,
        default='cell_line',
        choices=[
            'cell_line', 'cell_line_leaveout', 'drug_combo_leaveout',
            'drug_leaveout', 'global',
            '1', '2', '3'
        ],
        help='数据划分方式：也可用数字 1/2/3（1=cell_line, 2=cell_line_leaveout, 3=drug_combo_leaveout）'
    )
    # 更严格的留出细胞系：valid 也按 cell-line 级留出，保证 train/valid/test 细胞系不重叠
    p.add_argument(
        '--cell_line_leaveout_strict_valid',
        type=int,
        default=1,
        help='1: 严格(valid按细胞系留出)；0: 旧逻辑(valid样本级分层切分)'
    )

    # -----------------------------
    #  早停（Early Stopping）
    # -----------------------------
    # patience：验证指标连续多少个 epoch 没提升就停止
    p.add_argument('--early_stop_patience', type=int, default=200, help='早停耐心值（patience）')
    # 监控的验证指标（与 PRMHSyn.py 中 metric_map 的 key 对齐）
    p.add_argument(
        '--early_stop_monitor',
        type=str,
        default='val_auc',
        choices=['val_auc', 'val_aupr', 'val_f1', 'val_acc', 'val_precision', 'val_recall', 'val_loss'],
        help='早停监控的验证指标'
    )
    # higher: 指标越大越好；lower: 指标越小越好（例如 val_loss）
    p.add_argument(
        '--early_stop_mode',
        type=str,
        default='higher',
        choices=['higher', 'lower'],
        help='早停模式：higher/lower'
    )
    # min_delta：认为“有提升”的最小改变量
    p.add_argument('--early_stop_min_delta', type=float, default=0.0, help='早停最小提升阈值（min_delta）')

    # -----------------------------
    #  学习率调度（LR Scheduler）
    # -----------------------------
    # none/cosine/step/reduce_on_plateau（与 PRMHSyn.py 的分支对齐）
    p.add_argument(
        '--lr_scheduler',
        type=str,
        default='none',
        choices=['none', 'cosine', 'step', 'reduce_on_plateau'],
        help='学习率调度策略'
    )
    p.add_argument('--lr_factor', type=float, default=0.5, help='学习率衰减系数（ReduceLROnPlateau/Step）')
    p.add_argument('--lr_patience', type=int, default=20, help='ReduceLROnPlateau 的 patience')
    p.add_argument('--lr_min', type=float, default=1e-6, help='学习率下限（eta_min/min_lr）')

    # -----------------------------
    #  训练细节
    # -----------------------------
    p.add_argument('--grad_clip', type=float, default=0.0, help='梯度裁剪阈值（0表示不裁剪）')
    # 是否使用 LayerNorm（PRMHSyn.py 中会读取 args.use_layer_norm）
    p.add_argument('--use_layer_norm', type=int, default=1, help='1使用LayerNorm；0不使用')

    # 是否使用 PageRank（消融开关）
    p.add_argument('--use_pagerank', type=int, default=1, help='1使用PageRank；0不使用（消融）')

    # 是否使用 SE 模块（消融开关）
    p.add_argument('--use_se', type=int, default=1, help='1使用SE模块；0不使用（消融）')

    args = p.parse_args()
    
    args.data_file = data_file
    args.device = device
    args.Use_Cell_Line_Feature = Use_Cell_Line_Feature
    # 兼容 int -> bool 的配置
    args.cell_line_leaveout_strict_valid = bool(int(args.cell_line_leaveout_strict_valid))
    args.use_layer_norm = bool(int(args.use_layer_norm))
    args.use_pagerank = bool(int(args.use_pagerank))
    args.use_se = bool(int(args.use_se))

    return args
