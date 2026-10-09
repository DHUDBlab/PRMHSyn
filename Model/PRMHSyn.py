# -*- coding: utf-8 -*-
"""
PRMHSyn (clean, no SSL)
Created on Tue Nov 14 09:28:11 2023
@author: Mengjie Chen
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import numpy as np
import Data_Process
import Synergy_Models
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
import Config
import time
import os.path as osp

args = Config.parse()

# -----------------------------
#      EarlyStopping Class
# -----------------------------
class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    
    Args:
        patience (int): Number of epochs to wait before stopping if no improvement
        mode (str): 'higher' or 'lower', whether higher or lower metric is better
        min_delta (float): Minimum change to qualify as an improvement
        checkpoint_dir (str): Directory to save model checkpoints
        verbose (bool): Whether to print early stopping messages
    """
    def __init__(self, patience=200, mode='higher', min_delta=0.0, checkpoint_dir='checkpoints', verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        # Initialize best score based on mode
        if mode == 'higher':
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + min_delta
        else:  # mode == 'lower'
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - min_delta
        
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.checkpoint_path = None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def step(self, metric, model, epoch=None):
        """
        Check if training should stop based on current metric.
        
        Args:
            metric (float): Current validation metric value
            model (nn.Module): Model to save if metric improves
            epoch (int, optional): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.is_better(metric, self.best_score):
            # Metric improved, save model and reset counter
            self.best_score = metric
            self.best_epoch = epoch if epoch is not None else self.counter
            self.counter = 0
            
            # Save checkpoint
            self.checkpoint_path = osp.join(self.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score,
            }, self.checkpoint_path)
            
            if self.verbose:
                print(f'Validation metric improved to {metric:.6f}. Model saved to {self.checkpoint_path}')
        else:
            # Metric did not improve
            self.counter += 1
            if self.verbose and epoch is not None and epoch % 10 == 0:
                print(f'Validation metric did not improve. Counter: {self.counter}/{self.patience}')
            
            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('=' * 50)
                    print(f'Early stopping triggered!')
                    print(f'Best epoch: {self.best_epoch}, Best score: {self.best_score:.6f}')
                    print(f'No improvement for {self.patience} consecutive epochs')
                    print('=' * 50)
        
        return self.early_stop
    
    def load_checkpoint(self, model):
        """
        Load the best model checkpoint.
        
        Args:
            model (nn.Module): Model to load checkpoint into
            
        Returns:
            dict: Checkpoint dictionary with epoch and best_score
        """
        if self.checkpoint_path is None or not osp.exists(self.checkpoint_path):
            if self.verbose:
                print(f'Warning: No checkpoint found at {self.checkpoint_path}')
            return None
        
        checkpoint = torch.load(self.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.verbose:
            print(f'Loaded best model from epoch {checkpoint["epoch"]} with score {checkpoint["best_score"]:.6f}')
        
        return checkpoint


# -----------------------------
#        Metrics & threshold search
# -----------------------------
def metrics(labels, predictions, epoch, type, threshold=0.5):
    """
    Calculate and log evaluation metrics.

    Args:
        labels: list/array of 0/1 labels
        predictions: list/array of probabilities (after sigmoid)
        epoch: current epoch (for logging)
        type: 'train' / 'valid' / 'test'
        threshold: decision threshold for converting probabilities to 0/1
    """
    try:
        labels_arr = np.array(labels).astype(int)
        preds_arr = np.array(predictions, dtype=float)

        # AUC / AUPR 使用概率，不依赖 threshold
        auc = roc_auc_score(labels_arr, preds_arr) 
        aupr = average_precision_score(labels_arr, preds_arr)

        # 其余指标基于二值化结果
        binary_predictions = (preds_arr >= threshold).astype(int)
        binary_labels = labels_arr

        f1 = f1_score(binary_labels, binary_predictions)
        accuracy = accuracy_score(binary_labels, binary_predictions)
        precision = precision_score(binary_labels, binary_predictions, zero_division=0)
        recall = recall_score(binary_labels, binary_predictions, zero_division=0)
        bacc = balanced_accuracy_score(binary_labels, binary_predictions)
        mcc = matthews_corrcoef(binary_labels, binary_predictions)
        kappa = cohen_kappa_score(binary_labels, binary_predictions)

        return auc, aupr, f1, accuracy, precision, recall, bacc, mcc, kappa
    except Exception as e:
        print(f"Error calculating metrics for {type}: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def find_best_threshold(labels, predictions, metric="f1", num_thresholds=200):
    """
    在验证集上搜索最佳阈值（默认 F1 最大），用于之后在测试集上评估。

    Args:
        labels: list/array of 0/1 labels
        predictions: list/array of probabilities (after sigmoid)
        metric: 'f1' or 'mcc' 作为搜索目标
        num_thresholds: 在 0~1 之间尝试的阈值个数

    Returns:
        best_thr: float, 最优阈值
        best_score: float, 在验证集上的最优分数
    """
    labels_arr = np.array(labels).astype(int)
    preds_arr = np.array(predictions, dtype=float)

    # 如果验证集只有一个类别，阈值搜索没有意义，直接返回0.5
    if len(np.unique(labels_arr)) < 2:
        return 0.5, 0.0

    best_thr = 0.5
    best_score = -1.0

    for thr in np.linspace(0.01, 0.99, num_thresholds):
        bin_pred = (preds_arr >= thr).astype(int)
        try:
            if metric == "mcc":
                score = matthews_corrcoef(labels_arr, bin_pred)
            else:  # 默认 f1
                score = f1_score(labels_arr, bin_pred)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr, best_score

# -----------------------------
#           Test
# -----------------------------
def test(model, drug_features, cell_line_feature, disease_feature, data, edges, labels):
    model.eval()
    with torch.no_grad():
        preds, _, _, drug_emb, cell_emb = model(drug_features, cell_line_feature, disease_feature, edges)
        loss = 0
        pred = []
        real = []
        valid_cell_count = 0
        for idx in range(data.numDrug, data.numDrug + data.CellsCount):
            # 跳过没有数据的细胞系
            if idx not in preds or len(preds[idx]) == 0:
                continue
            if idx not in labels or len(labels[idx]) == 0:
                continue
            
            pred.extend(torch.sigmoid(preds[idx]).cpu().detach().numpy())
            real.extend(labels[idx].cpu().detach().numpy())
            each_preds = preds[idx]
            each_labels = labels[idx]
            each_pos_weight = data.pos_weights[idx]
            criterion = nn.BCEWithLogitsLoss(pos_weight=each_pos_weight)
            each_loss = criterion(each_preds, each_labels)
            loss += each_loss
            valid_cell_count += 1
        if valid_cell_count > 0:
            loss = loss / valid_cell_count
        else:
            loss = torch.tensor(0.0, device=loss.device)
    return loss, pred, real 


# -----------------------------
#   Feature extraction for visualization
# -----------------------------
def extract_joint_features(
    model,
    drug_features,
    cell_line_feature,
    disease_feature,
    data,
    edges,
    labels,
    file_prefix="joint_features",
):
    """
    Extract joint features (before and after model encoding) on a given split
    for dimensionality reduction visualization (e.g., t-SNE / UMAP).

    Joint raw feature for a sample (drugA, drugB, cell):
        [Drug_Features[drugA] || Drug_Features[drugB] || Cell_Line_Feature[cell]]

    Joint processed feature:
        [drug_emb[drugA] || drug_emb[drugB] || cell_emb[cell]]
    """
    model.eval()
    all_raw = []
    all_proc = []
    all_y = []

    with torch.no_grad():
        # preds is not used here; we only need embeddings
        preds, _, _, drug_emb, cell_emb = model(
            drug_features, cell_line_feature, disease_feature, edges
        )

        # drug_features / cell_line_feature are tensors on device
        # embeddings are also tensors on device
        for idx in range(data.numDrug, data.numDrug + data.CellsCount):
            if idx not in edges or len(edges[idx]) == 0:
                continue
            if idx not in labels or len(labels[idx]) == 0:
                continue

            cell_local_id = idx - data.numDrug
            cell_raw = cell_line_feature[cell_local_id]
            cell_enc = cell_emb[cell_local_id]

            edge_list = edges[idx]
            label_list = labels[idx]

            # ensure we iterate over pairs of (drugA, drugB, y)
            for (a, b), y in zip(edge_list, label_list):
                a = int(a)
                b = int(b)
                y = int(y)

                drugA_raw = drug_features[a]
                drugB_raw = drug_features[b]
                drugA_enc = drug_emb[a]
                drugB_enc = drug_emb[b]

                joint_raw = torch.cat(
                    [drugA_raw, drugB_raw, cell_raw], dim=0
                ).detach().cpu().numpy()
                joint_proc = torch.cat(
                    [drugA_enc, drugB_enc, cell_enc], dim=0
                ).detach().cpu().numpy()

                all_raw.append(joint_raw)
                all_proc.append(joint_proc)
                all_y.append(y)

    if len(all_raw) == 0:
        print(f"[FeatureExtract] No samples found for {file_prefix}, skip saving.")
        return

    all_raw = np.asarray(all_raw)
    all_proc = np.asarray(all_proc)
    all_y = np.asarray(all_y, dtype=int)

    np.save(f"{file_prefix}_raw.npy", all_raw)
    np.save(f"{file_prefix}_proc.npy", all_proc)
    np.save(f"{file_prefix}_labels.npy", all_y)
    print(
        f"[FeatureExtract] Saved: {file_prefix}_raw.npy, "
        f"{file_prefix}_proc.npy, {file_prefix}_labels.npy "
        f"(num_samples={len(all_y)})"
    )


# -----------------------------
#           Train
# -----------------------------
def train(model, drug_features, cell_line_feature, disease_feature,
          data, optimizer, epochs, fold_idx=None, scheduler=None, grad_clip=0.0):
    """
    Train PRMHSyn for given number of epochs.
    仅监督任务：各细胞系上的 BCEWithLogits（带 pos_weight）。
    支持早停机制（Early Stopping）。
    """
    model.train()
    
    # Create EarlyStopping object
    checkpoint_dir = osp.join('checkpoints', f'fold_{fold_idx}' if fold_idx is not None else 'default')
    stopper = EarlyStopping(
        patience=args.early_stop_patience,
        mode=args.early_stop_mode,
        min_delta=args.early_stop_min_delta,
        checkpoint_dir=checkpoint_dir,
        verbose=False
    )

    for epoch in range(epochs):
        preds, _, _, _, _ = model(
            drug_features, cell_line_feature, disease_feature, data.train_edges
        )

        loss_train = 0
        train_pred = []
        train_real = []

        # -------- Supervised BCE loss --------
        for idx in range(data.numDrug, data.numDrug + data.CellsCount):
            # 跳过没有训练数据的细胞系
            if idx not in preds or len(preds[idx]) == 0:
                continue
            if idx not in data.train_labels or len(data.train_labels[idx]) == 0:
                continue
            
            train_real.extend(data.train_labels[idx].cpu().detach().numpy())
            each_labels = data.train_labels[idx]
            # 确保preds[idx]的长度是each_labels的两倍（因为有反向边）
            if len(preds[idx]) != len(each_labels) * 2:
                # 如果长度不匹配，尝试只使用前半部分
                if len(preds[idx]) >= len(each_labels):
                    each_preds = preds[idx][:len(each_labels)]
                else:
                    continue  # 跳过这个细胞系
            else:
                each_preds = (torch.add(preds[idx][:len(each_labels)], preds[idx][len(each_labels):])) / 2
            train_pred.extend(torch.sigmoid(each_preds).cpu().detach().numpy())
            each_pos_weight = data.pos_weights[idx]
            criterion = nn.BCEWithLogitsLoss(pos_weight=each_pos_weight)
            each_loss = criterion(each_preds, each_labels)
            loss_train += each_loss

        sup_loss = loss_train / data.CellsCount
        total_loss = sup_loss

        # L2 regularization is handled by optimizer's weight_decay parameter

        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent gradient explosion
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Learning rate scheduling (for non-plateau schedulers)
        if scheduler is not None and args.lr_scheduler != 'reduce_on_plateau':
            scheduler.step()

        # -------- Logging --------（训练阶段仍使用固定阈值0.5，仅用于监控）
        train_results = metrics(train_real, train_pred, epoch, 'train', threshold=0.5)

        loss_valid, valid_pred, valid_real = test(
            model, drug_features, cell_line_feature, disease_feature,
            data, data.valid_edges, data.valid_labels
        )
        valid_results = metrics(valid_real, valid_pred, epoch, 'valid', threshold=0.5)

        # Get the monitored metric value
        # valid_results: (auc, aupr, f1, accuracy, precision, recall) = (0, 1, 2, 3, 4, 5)
        metric_map = {
            'val_auc': valid_results[0],      # AUC-ROC (index 0)
            'val_aupr': valid_results[1],     # AUC-PR (index 1)
            'val_f1': valid_results[2],       # F1 score (index 2)
            'val_acc': valid_results[3],      # Accuracy (index 3)
            'val_precision': valid_results[4], # Precision (index 4)
            'val_recall': valid_results[5],   # Recall (index 5)
            'val_loss': float(loss_valid)     # Loss
        }
        current_score = metric_map[args.early_stop_monitor]
        
        # Learning rate scheduling for ReduceLROnPlateau
        if scheduler is not None and args.lr_scheduler == 'reduce_on_plateau':
            scheduler.step(current_score)
        
        # Use EarlyStopping: step() returns True if should stop
        # When validation metric improves, model is automatically saved
        # When validation metric doesn't improve, counter increments
        # If counter >= patience, early stopping is triggered
        should_stop = stopper.step(current_score, model, epoch=epoch)
        
        # Early stopping check - if patience exceeded, stop training
        if should_stop:
            break

        if epoch % 20 == 0:
            print('Epoch: ', epoch,
                  'TrainLoss: {:.6f},'.format(total_loss),
                  'AUC-ROC: {:.6f},'.format(train_results[0]),
                  'AUC-PR: {:.6f},'.format(train_results[1]),
                  'F1: {:.6f},'.format(train_results[2]),
                  'ACC: {:.6f},'.format(train_results[3]),
                  'Precision: {:.6f},'.format(train_results[4]),
                  'Recall: {:.6f},'.format(train_results[5]),
                  'BACC: {:.6f},'.format(train_results[6]),
                  'MCC: {:.6f},'.format(train_results[7]),
                  'Kappa: {:.6f}'.format(train_results[8]))
            print('Epoch: ', epoch,
                  'ValLoss: {:.6f},'.format(loss_valid),
                  'AUC-ROC: {:.6f},'.format(valid_results[0]),
                  'AUC-PR: {:.6f},'.format(valid_results[1]),
                  'F1: {:.6f},'.format(valid_results[2]),
                  'ACC: {:.6f},'.format(valid_results[3]),
                  'Precision: {:.6f},'.format(valid_results[4]),
                  'Recall: {:.6f},'.format(valid_results[5]),
                  'BACC: {:.6f},'.format(valid_results[6]),
                  'MCC: {:.6f},'.format(valid_results[7]),
                  'Kappa: {:.6f}'.format(valid_results[8]))
            print(f'Best {args.early_stop_monitor}: {stopper.best_score:.6f} (epoch {stopper.best_epoch}), Patience: {stopper.counter}/{stopper.patience}')

    # Final summary
    if not stopper.early_stop:
        print('=' * 50)
        print(f'Training completed: {epochs} epochs')
        print(f'Best epoch: {stopper.best_epoch}, Best {args.early_stop_monitor}: {stopper.best_score:.6f}')
        print('=' * 50)
        pass
    
    # Load the best model checkpoint
    checkpoint = stopper.load_checkpoint(model)
    
    return model, stopper.best_epoch


# -----------------------------
#             Main
# -----------------------------
def run_training():
    """完整 K 折训练与汇总；由本文件以脚本方式运行时调用（见末尾 if __name__）。"""
    start_time = time.time()
    print(f"Dataset: {args.dataset}")
    # 静默：不使用 MLflow 记录实验

    # 精简版数据加载：仅使用训练所需的核心数据
    realFolds, Drug_Features, Cell_Line_Feature, _, _, disease_features, smiles_list = Data_Process.process_data(args.dataset)

    # ---------- Disease features ----------
    if disease_features is not None:
        Disease_Feature_all = torch.tensor(disease_features).float().to(args.device)
    else:
        try:
            disease_emb_np = np.load(osp.join(args.data_file, 'DISEASE', 'DISEASE_EMBEDDING_BioBERT.npy'))
            Disease_Feature_all = torch.tensor(disease_emb_np).float().to(args.device)
            print("使用原始疾病嵌入特征")
        except Exception:
            Disease_Feature_all = torch.zeros((1, 768)).float().to(args.device)
            print("使用默认疾病特征")

    # ---------- Protein text features ----------
    feat_path = osp.join(args.data_file, 'PROTEIN', 'protein_text_features.npy')
    alt_feat_path = osp.join(args.data_file, 'PROTEIN', 'relevant_protein_text_features.npy')
    if os.path.exists(feat_path):
        protein_text_features = np.load(feat_path)
    elif os.path.exists(alt_feat_path):
        protein_text_features = np.load(alt_feat_path)
    else:
        protein_text_features = None
    
    # Normalize protein features (Z-score standardization, consistent with Drug_Features and Cell_Line_Feature)
    if protein_text_features is not None:
        protein_mean = np.mean(protein_text_features, axis=0)
        protein_std = np.std(protein_text_features, axis=0)
        protein_std[protein_std == 0] = 1.0  # Avoid division by zero
        protein_text_features = (protein_text_features - protein_mean) / protein_std
        # 静默

    Drug_Features = torch.tensor(Drug_Features).float().to(args.device)
    Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(args.device)

    results = []  # 存储最优阈值下的结果
    results_thr05 = []  # 存储阈值0.5下的结果

    # basic sanity checks
    if args.epochs <= 0:
        raise ValueError("Epochs must be positive")
    if args.k_fold <= 0:
        raise ValueError("K-fold must be positive")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    # 静默：避免运行时刷屏

    # ---------- K-fold cross-validation ----------
    for fold in range(args.k_fold):
        try:
            torch.manual_seed(args.seed)

            Data = realFolds[fold]
            Data = Data_Process.torch_from_numpy(Data, args.device)

            # 静默

            # Data.numNode = numDrug + numCellline + numDisease + numProtein
            cur_num_disease = Data.numNode - (Data.numDrug + Data.CellsCount)
            # 假设疾病与蛋白质数量相等（可根据实际情况调整）
            cur_num_protein = cur_num_disease // 2 if cur_num_disease > 0 else 0
            cur_num_disease = cur_num_disease - cur_num_protein

            # 静默

            # 当前 fold 的 Disease 特征
            if cur_num_disease > 0:
                if Disease_Feature_all.shape[0] >= cur_num_disease:
                    Disease_Feature = Disease_Feature_all[:cur_num_disease, :]
                else:
                    pad = torch.zeros(
                        cur_num_disease - Disease_Feature_all.shape[0],
                        Disease_Feature_all.shape[1],
                        device=args.device
                    )
                    Disease_Feature = torch.cat([Disease_Feature_all, pad], dim=0)
            else:
                Disease_Feature = torch.zeros(
                    0, Disease_Feature_all.shape[1], device=args.device
                )

            # 当前 fold 的 protein 特征
            if protein_text_features is not None and cur_num_protein > 0:
                if protein_text_features.shape[0] >= cur_num_protein:
                    protein_features = protein_text_features[:cur_num_protein, :]
                else:
                    pad = np.zeros(
                        (cur_num_protein - protein_text_features.shape[0],
                         protein_text_features.shape[1])
                    )
                    protein_features = np.vstack([protein_text_features, pad])
            else:
                protein_features = None

            # ---------- Model ----------
            # LayerNorm 由 Config 的 --use_layer_norm 控制（默认 1）
            use_layer_norm = args.use_layer_norm
            # 静默
            
            Model = Synergy_Models.PRMHSyn(
                Data.numDrug,
                Synergy_Models.BioEncoder(
                    Drug_Features.shape[1],
                    Cell_Line_Feature.shape[1],
                    Disease_Feature.shape[1] if Disease_Feature.shape[0] > 0
                    else Disease_Feature_all.shape[1],
                    Data.CellsCount,
                    cur_num_protein,
                    512,
                    device=args.device,
                    protein_features=protein_features,
                    smiles_list=smiles_list,
                    # 完全恢复原始设置：使用分子图GCN编码 + Highway 与传统特征融合
                    use_molecular_gcn=True,
                    use_layer_norm=use_layer_norm
                ),
                Synergy_Models.RHGNN(
                    Data.V,
                    Data.E,
                    Data.hypergraph_edge_num,
                    Data.edge_length,
                    Data.degV_dict,
                    512,
                    128,
                    256,
                    num_edge_types=4,
                    dropout=0.0,
                    hyperedges=[],
                    hyperedge_types=[],
                    use_layer_norm=use_layer_norm,
                    use_pagerank=args.use_pagerank,
                    use_se=args.use_se
                ),
                Synergy_Models.CrossGatedFiLM_SynergyDecoder(
                    emb_dim=256,
                    hidden_dim=256,
                    output_dim=1,
                    dropout=0.1,  # 恢复到0.1 
                    numDrug=Data.numDrug,
                    use_layer_norm=use_layer_norm
                )
            ).to(args.device)

            optimizer = torch.optim.Adam(
                list(Model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay  # L2 regularization via optimizer's weight_decay
            )
            
            # Learning rate scheduler
            if args.lr_scheduler == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max' if args.early_stop_mode == 'higher' else 'min',
                    factor=args.lr_factor, patience=args.lr_patience,
                    min_lr=args.lr_min, verbose=False
                )
            elif args.lr_scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs, eta_min=args.lr_min
                )
            elif args.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=args.epochs // 3, gamma=args.lr_factor
                )
            else:
                scheduler = None


            # ---------- Train ----------
            # Train function now returns the model (with best checkpoint loaded) and best_epoch
            Model, best_epoch = train(
                Model,
                Drug_Features,
                Cell_Line_Feature,
                Disease_Feature,
                Data,
                optimizer,
                args.epochs,
                fold_idx=fold,
                scheduler=scheduler,
                grad_clip=args.grad_clip if hasattr(args, 'grad_clip') else 0.0
            )

            # Model already has best checkpoint loaded by EarlyStopping.load_checkpoint()
            # 不保存 MLflow 模型

            # ---------- Threshold selection on validation set ----------
            # 使用最佳模型在验证集上搜索“最优阈值”（默认 F1 最大）
            loss_valid_final, valid_pred_final, valid_real_final = test(
                Model,
                Drug_Features,
                Cell_Line_Feature,
                Disease_Feature,
                Data,
                Data.valid_edges,
                Data.valid_labels
            )
            best_threshold, best_f1_valid = find_best_threshold(valid_real_final, valid_pred_final, metric="f1")

            # ---------- Test ----------
            loss_test, test_pred, test_real = test(
                Model,
                Drug_Features,
                Cell_Line_Feature,
                Disease_Feature,
                Data,
                Data.test_edges,
                Data.test_labels
            )

            # ---------- Feature extraction for visualization (default: on test set) ----------
            # 导出“训练前原始联合特征”和“训练后联合嵌入特征”用于降维可视化
            feature_prefix = f"joint_features_{args.dataset}_fold{fold + 1}"
            extract_joint_features(
                Model,
                Drug_Features,
                Cell_Line_Feature,
                Disease_Feature,
                Data,
                Data.test_edges,
                Data.test_labels,
                file_prefix=feature_prefix,
            )

            # 1) 在测试集上使用固定阈值 0.5 计算一套指标（便于对比）
            test_results_thr05 = metrics(test_real, test_pred, best_epoch, 'test_thr0.5', threshold=0.5)
            results_thr05.append(list(test_results_thr05))

            # 2) 在测试集上使用验证集选出的最优阈值计算一套指标（当前主结果）
            test_results_best = metrics(test_real, test_pred, best_epoch, 'test_best', threshold=best_threshold)
            results.append(list(test_results_best))

            # 控制台打印两套结果
            print(f'Fold {fold + 1}/{args.k_fold} - threshold=0.5 - '
                  f'loss_test: {loss_test:.6f}, '
                  f'AUC-ROC: {test_results_thr05[0]:.6f}, '
                  f'AUC-PR: {test_results_thr05[1]:.6f}, '
                  f'F1: {test_results_thr05[2]:.6f}, '
                  f'ACC: {test_results_thr05[3]:.6f}, '
                  f'Precision: {test_results_thr05[4]:.6f}, '
                  f'Recall: {test_results_thr05[5]:.6f}, '
                  f'BACC: {test_results_thr05[6]:.6f}, '
                  f'MCC: {test_results_thr05[7]:.6f}, '
                  f'Kappa: {test_results_thr05[8]:.6f}')

            print(f'Fold {fold + 1}/{args.k_fold} - threshold=best({best_threshold:.4f}) - '
                  f'loss_test: {loss_test:.6f}, '
                  f'AUC-ROC: {test_results_best[0]:.6f}, '
                  f'AUC-PR: {test_results_best[1]:.6f}, '
                  f'F1: {test_results_best[2]:.6f}, '
                  f'ACC: {test_results_best[3]:.6f}, '
                  f'Precision: {test_results_best[4]:.6f}, '
                  f'Recall: {test_results_best[5]:.6f}, '
                  f'BACC: {test_results_best[6]:.6f}, '
                  f'MCC: {test_results_best[7]:.6f}, '
                  f'Kappa: {test_results_best[8]:.6f}')
            print(f'Best epoch: {best_epoch}')

            # 仍然用“best 阈值下的 AUPR”作为整体结果的度量
            pass

        except Exception as e:
            print(f"Error in fold {fold + 1}: {str(e)}")

        finally:
            if 'Model' in locals():
                del Model
            if 'optimizer' in locals():
                del optimizer
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

            pass

    # ---------- Final summary ----------
    if results:
        results = pd.DataFrame(results).to_numpy()
        mean_results = np.mean(results, axis=0)
        std_results = np.std(results, axis=0)

        # 阈值0.5的结果统计
        if results_thr05:
            results_thr05 = pd.DataFrame(results_thr05).to_numpy()
            mean_results_thr05 = np.mean(results_thr05, axis=0)
            std_results_thr05 = np.std(results_thr05, axis=0)
        else:
            mean_results_thr05 = None
            std_results_thr05 = None

        print('=' * 50)
        print('FINAL RESULTS (threshold=0.5):')
        print('=' * 50)
        if mean_results_thr05 is not None:
            print(f'Mean - AUC-ROC: {mean_results_thr05[0]:.6f}, '
                  f'AUC-PR: {mean_results_thr05[1]:.6f}, '
                  f'F1: {mean_results_thr05[2]:.6f}, '
                  f'ACC: {mean_results_thr05[3]:.6f}, '
                  f'Precision: {mean_results_thr05[4]:.6f}, '
                  f'Recall: {mean_results_thr05[5]:.6f}, '
                  f'BACC: {mean_results_thr05[6]:.6f}, '
                  f'MCC: {mean_results_thr05[7]:.6f}, '
                  f'Kappa: {mean_results_thr05[8]:.6f}')
            print(f'Std  - AUC-ROC: {std_results_thr05[0]:.6f}, '
                  f'AUC-PR: {std_results_thr05[1]:.6f}, '
                  f'F1: {std_results_thr05[2]:.6f}, '
                  f'ACC: {std_results_thr05[3]:.6f}, '
                  f'Precision: {std_results_thr05[4]:.6f}, '
                  f'Recall: {std_results_thr05[5]:.6f}, '
                  f'BACC: {std_results_thr05[6]:.6f}, '
                  f'MCC: {std_results_thr05[7]:.6f}, '
                  f'Kappa: {std_results_thr05[8]:.6f}')

        print('=' * 50)
        print('FINAL RESULTS (threshold=best from validation):')
        print('=' * 50)
        print(f'Mean - AUC-ROC: {mean_results[0]:.6f}, '
              f'AUC-PR: {mean_results[1]:.6f}, '
              f'F1: {mean_results[2]:.6f}, '
              f'ACC: {mean_results[3]:.6f}, '
              f'Precision: {mean_results[4]:.6f}, '
              f'Recall: {mean_results[5]:.6f}, '
              f'BACC: {mean_results[6]:.6f}, '
              f'MCC: {mean_results[7]:.6f}, '
              f'Kappa: {mean_results[8]:.6f}')
        print(f'Std  - AUC-ROC: {std_results[0]:.6f}, '
              f'AUC-PR: {std_results[1]:.6f}, '
              f'F1: {std_results[2]:.6f}, '
              f'ACC: {std_results[3]:.6f}, '
              f'Precision: {std_results[4]:.6f}, '
              f'Recall: {std_results[5]:.6f}, '
              f'BACC: {std_results[6]:.6f}, '
              f'MCC: {std_results[7]:.6f}, '
              f'Kappa: {std_results[8]:.6f}')
        print('=' * 50)

        output_file = f"output_PRMHSyn_{args.dataset}.txt"
        with open(output_file, "w") as file:
            file.write(f"Dataset: {args.dataset}\n")
            file.write(f"Learning Rate: {args.learning_rate}\n")
            file.write(f"Weight Decay: {args.weight_decay}\n")
            file.write(f"Epochs: {args.epochs}\n")
            file.write(f"K-Fold: {args.k_fold}\n")
            file.write("=" * 50 + "\n")
            file.write("Results with threshold=0.5:\n")
            file.write("-" * 50 + "\n")
            if mean_results_thr05 is not None:
                for i, item in enumerate(results_thr05):
                    file.write(f"Fold {i + 1}: {item}\n")
                file.write("-" * 50 + "\n")
                file.write(f"Mean: {mean_results_thr05}\n")
                file.write(f"Std: {std_results_thr05}\n")
            file.write("=" * 50 + "\n")
            file.write("Results with threshold=best (from validation):\n")
            file.write("-" * 50 + "\n")
            for i, item in enumerate(results):
                file.write(f"Fold {i + 1}: {item}\n")
            file.write("-" * 50 + "\n")
            file.write(f"Mean: {mean_results}\n")
            file.write(f"Std: {std_results}\n")

        print(f"Results saved to: {output_file}")

        # 不使用 MLflow 记录结果
    else:
        print("No results obtained. Check for errors in training.")


if __name__ == '__main__':
    run_training()
