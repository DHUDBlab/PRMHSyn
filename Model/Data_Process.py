import Config
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
import random
import torch
import torch_sparse

params = Config.parse()

# 关闭 RDKit 的 warning 日志（避免 MorganGenerator 弃用提示刷屏）
RDLogger.DisableLog('rdApp.warning')

def Incidence_matrix(edge_list, num_node):
    # Construct sparse matrix
    row_indices = []
    col_indices = []
    values = []
    for index in range(len(edge_list)):
        for i, drug_id in enumerate(edge_list[index]):
            row_indices.append(drug_id)
            col_indices.append(index)
            values.append(1)
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_node, len(edge_list)))
    return sparse_matrix

def Hyperedge_Index(Hyperedges):
    hyperedge_index = [[],[]]
    for i in range(len(Hyperedges)):
        length = len(Hyperedges[i])
        hyperedge_index[0].extend(Hyperedges[i])
        hyperedge_index[1].extend([i] * length)
    return hyperedge_index

def Construct_Hypergraph(hyperedge_groups, node_num):
    # hyperedge_groups: List[List[List[int]]] 多类超边，每一类是若干超边
    G = []
    degV = {}
    for i, g in enumerate(hyperedge_groups):
        G += g
        h = Incidence_matrix(g, node_num)
        degv = torch.from_numpy(h.sum(1)).view(-1, 1).float().pow(-1)
        degv[degv.isinf()] = 1
        degV[i] = degv.numpy()
    H = Incidence_matrix(G, node_num)
    (row, col), _ = torch_sparse.from_scipy(H)
    V, E = row, col
    return V.numpy(), E.numpy(), degV


def _split_method_id(method):
    m = str(method).strip().lower()
    if m in ("1", "cell_line", "cellline"):
        return 1
    if m in ("2", "cell_line_leaveout", "cellline_leaveout"):
        return 2
    if m in ("3", "drug_combo_leaveout", "drugcombo_leaveout", "drug_combo"):
        return 3
    return 1


def produce_cell_line_leaveout(data, drug_disease_pairs=None, disease_offset=None, dti_edges=None, numProtein=0, disease_features=None, num_disease=0):
    Interaction_Score, numDrug, numCellline, iFold = data
    Interaction_Score = Interaction_Score.copy()
    for ridx, row in Interaction_Score.iterrows():
        a, b = row['DrugA'], row['DrugB']
        Interaction_Score.at[ridx, 'DrugA'], Interaction_Score.at[ridx, 'DrugB'] = sorted([a, b])

    samples = []
    ys = []
    for _, row in Interaction_Score.iterrows():
        score = row.iloc[3]
        if score >= params.threshold:
            y = 1
        elif score < 0:
            y = 0
        else:
            continue
        samples.append((int(row['DrugA']), int(row['DrugB']), int(row['CellLine']), int(y)))
        ys.append(int(y))

    train_edges, train_labels, valid_edges, valid_labels, test_edges, test_labels, pos_weights = {}, {}, {}, {}, {}, {}, {}
    all_cell_lines = list(range(numDrug, numDrug + numCellline))
    for c in all_cell_lines:
        train_edges[c], valid_edges[c], test_edges[c] = [], [], []
        train_labels[c], valid_labels[c], test_labels[c] = [], [], []
        pos_weights[c] = 1.0

    if len(samples) == 0:
        Synergistic_Graph, Antagonistic_Graph = [], []
        Synergistic_Types, Antagonistic_Types = [], []
        hyperedges, hyperedge_types = [], []
        drug_disease_edges = []
    else:
        from sklearn.model_selection import KFold, StratifiedShuffleSplit

        all_cells = sorted({cell for _, _, cell, _ in samples})
        kf = KFold(n_splits=params.k_fold, shuffle=True, random_state=params.seed)
        cell_folds = list(kf.split(np.zeros(len(all_cells))))
        train_cells_idx, test_cells_idx = cell_folds[iFold]
        test_cell_set = {all_cells[i] for i in test_cells_idx}

        test_idx = [i for i, (_, _, cell, _) in enumerate(samples) if cell in test_cell_set]
        trainval_idx = [i for i, (_, _, cell, _) in enumerate(samples) if cell not in test_cell_set]

        strict_valid = getattr(params, 'cell_line_leaveout_strict_valid', True)
        if strict_valid:
            trainval_cells = sorted({samples[i][2] for i in trainval_idx})
            if len(trainval_cells) < 2:
                valid_idx = np.array([], dtype=int)
                train_idx = np.array(trainval_idx, dtype=int)
            else:
                inner_k = min(int(params.k_fold), len(trainval_cells))
                inner_kf = KFold(n_splits=inner_k, shuffle=True, random_state=int(params.seed) + 12345)
                inner_folds = list(inner_kf.split(np.zeros(len(trainval_cells))))
                inner_fold_id = int(iFold) % len(inner_folds)
                _, inner_valid_cells_idx = inner_folds[inner_fold_id]
                valid_cell_set = {trainval_cells[i] for i in inner_valid_cells_idx}
                valid_idx = np.array([i for i in trainval_idx if samples[i][2] in valid_cell_set], dtype=int)
                train_idx = np.array([i for i in trainval_idx if samples[i][2] not in valid_cell_set], dtype=int)
        else:
            valid_ratio = max(min(1.0 / float(params.k_fold), 0.5), 0.05)
            y_trainval = np.array([ys[i] for i in trainval_idx]) if len(trainval_idx) else np.array([])
            if len(trainval_idx) == 0 or len(np.unique(y_trainval)) < 2:
                valid_idx = np.array([], dtype=int)
                train_idx = np.array(trainval_idx, dtype=int)
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=params.seed + iFold)
                rel_train_idx, rel_valid_idx = next(sss.split(np.zeros(len(trainval_idx)), y_trainval))
                train_idx = np.array(trainval_idx, dtype=int)[rel_train_idx]
                valid_idx = np.array(trainval_idx, dtype=int)[rel_valid_idx]

        def _append(edges, labels, a, b, cell, y):
            edges[cell].append([a, b])
            labels[cell].append(int(y))

        for i in train_idx:
            a, b, cell, y = samples[i]
            _append(train_edges, train_labels, a, b, cell, y)
        for i in valid_idx:
            a, b, cell, y = samples[i]
            _append(valid_edges, valid_labels, a, b, cell, y)
        for i in test_idx:
            a, b, cell, y = samples[i]
            _append(test_edges, test_labels, a, b, cell, y)

        Synergistic_Graph, Antagonistic_Graph = [], []
        Synergistic_Types, Antagonistic_Types = [], []
        for c in all_cell_lines:
            cell_train = np.array(train_edges[c]).reshape(-1, 2) if len(train_edges[c]) else np.array([]).reshape(0, 2)
            if len(cell_train) > 0:
                cell_train = np.concatenate([cell_train, [[x[1], x[0]] for x in cell_train]])
            train_edges[c] = cell_train
            valid_edges[c] = np.array(valid_edges[c]).reshape(-1, 2) if len(valid_edges[c]) else np.array([]).reshape(0, 2)
            test_edges[c] = np.array(test_edges[c]).reshape(-1, 2) if len(test_edges[c]) else np.array([]).reshape(0, 2)

            y_cell_train = train_labels[c]
            pos_cnt = int(np.sum(np.array(y_cell_train) == 1)) if len(y_cell_train) else 0
            neg_cnt = int(np.sum(np.array(y_cell_train) == 0)) if len(y_cell_train) else 0
            pos_weights[c] = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0

            if len(train_labels[c]) > 0:
                orig_train = np.array(train_edges[c][:len(train_labels[c])]).reshape(-1, 2)
                for (a, b), y in zip(orig_train.tolist(), train_labels[c]):
                    if int(y) == 1:
                        Synergistic_Graph.append(sorted([int(a), int(b), int(c)]))
                        Synergistic_Types.append('synergistic')
                    else:
                        Antagonistic_Graph.append(sorted([int(a), int(b), int(c)]))
                        Antagonistic_Types.append('antagonistic')

        hyperedges = Synergistic_Graph + Antagonistic_Graph
        hyperedge_types = Synergistic_Types + Antagonistic_Types
        drug_disease_edges = []

    if drug_disease_pairs is not None and disease_offset is not None:
        for drug_id, disease_id in drug_disease_pairs:
            hyperedges.append([drug_id, disease_offset + disease_id])
            hyperedge_types.append('drug-disease')
        drug_disease_edges = [[int(d), int(disease_offset + dis)] for d, dis in drug_disease_pairs]
    if dti_edges is None:
        dti_edges = []

    if len(drug_disease_edges) > 0:
        node_num = disease_offset + num_disease + numProtein
    else:
        node_num = numDrug + numCellline + numProtein
        disease_offset = None

    hypergraph_groups = [Synergistic_Graph, Antagonistic_Graph, drug_disease_edges, dti_edges]
    hypergraph = Synergistic_Graph + Antagonistic_Graph + drug_disease_edges + dti_edges
    edge_length = [len(Synergistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph) + len(drug_disease_edges),
                   len(hypergraph)]
    V, E, degV = Construct_Hypergraph(hypergraph_groups, node_num)
    realFold = RealFoldData(train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights)
    realFold.iFold = iFold
    realFold.CellsCount = numCellline
    realFold.numDrug = numDrug
    realFold.numNode = node_num if len(drug_disease_edges) > 0 else (numDrug + numCellline)
    realFold.V = V
    realFold.E = E
    realFold.edge_length = edge_length
    realFold.degV_dict = degV
    realFold.hypergraph_edge_num = len(hypergraph)
    realFold.hyperedges = hyperedges
    realFold.hyperedge_types = hyperedge_types
    return realFold


def produce_drug_combo_leaveout(data, drug_disease_pairs=None, disease_offset=None, dti_edges=None, numProtein=0, disease_features=None, num_disease=0):
    Interaction_Score, numDrug, numCellline, iFold = data
    Interaction_Score = Interaction_Score.copy()
    for ridx, row in Interaction_Score.iterrows():
        a, b = row['DrugA'], row['DrugB']
        Interaction_Score.at[ridx, 'DrugA'], Interaction_Score.at[ridx, 'DrugB'] = sorted([a, b])

    samples, ys, pairs = [], [], []
    for _, row in Interaction_Score.iterrows():
        score = row.iloc[3]
        if score >= params.threshold:
            y = 1
        elif score < 0:
            y = 0
        else:
            continue
        a = int(row['DrugA']); b = int(row['DrugB']); c = int(row['CellLine'])
        samples.append((a, b, c, int(y)))
        ys.append(int(y))
        pairs.append((a, b))

    train_edges, train_labels, valid_edges, valid_labels, test_edges, test_labels, pos_weights = {}, {}, {}, {}, {}, {}, {}
    all_cell_lines = list(range(numDrug, numDrug + numCellline))
    for c in all_cell_lines:
        train_edges[c], valid_edges[c], test_edges[c] = [], [], []
        train_labels[c], valid_labels[c], test_labels[c] = [], [], []
        pos_weights[c] = 1.0

    if len(samples) == 0:
        Synergistic_Graph, Antagonistic_Graph = [], []
        Synergistic_Types, Antagonistic_Types = [], []
        hyperedges, hyperedge_types = [], []
        drug_disease_edges = []
    else:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold

        pair_to_indices = {}
        for i, p in enumerate(pairs):
            pair_to_indices.setdefault(p, []).append(i)
        uniq_pairs = list(pair_to_indices.keys())
        pair_labels = []
        for p in uniq_pairs:
            idxs = pair_to_indices[p]
            mean_y = float(np.mean([ys[j] for j in idxs])) if len(idxs) else 0.0
            pair_labels.append(1 if mean_y >= 0.5 else 0)

        if len(np.unique(pair_labels)) < 2:
            kf = KFold(n_splits=params.k_fold, shuffle=True, random_state=params.seed)
            folds = list(kf.split(np.zeros(len(uniq_pairs))))
        else:
            skf = StratifiedKFold(n_splits=params.k_fold, shuffle=True, random_state=params.seed)
            folds = list(skf.split(np.zeros(len(uniq_pairs)), np.array(pair_labels)))

        train_pairs_idx, test_pairs_idx = folds[iFold]
        test_pair_set = {uniq_pairs[i] for i in test_pairs_idx}

        test_idx, trainval_idx = [], []
        for p, idxs in pair_to_indices.items():
            (test_idx if p in test_pair_set else trainval_idx).extend(idxs)

        valid_ratio = max(min(1.0 / float(params.k_fold), 0.5), 0.05)
        y_trainval = np.array([ys[i] for i in trainval_idx]) if len(trainval_idx) else np.array([])
        if len(trainval_idx) == 0 or len(np.unique(y_trainval)) < 2:
            valid_idx = np.array([], dtype=int)
            train_idx = np.array(trainval_idx, dtype=int)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=params.seed + iFold)
            rel_train_idx, rel_valid_idx = next(sss.split(np.zeros(len(trainval_idx)), y_trainval))
            train_idx = np.array(trainval_idx, dtype=int)[rel_train_idx]
            valid_idx = np.array(trainval_idx, dtype=int)[rel_valid_idx]

        def _append(edges, labels, a, b, cell, y):
            edges[cell].append([a, b])
            labels[cell].append(int(y))

        for i in train_idx:
            a, b, cell, y = samples[i]
            _append(train_edges, train_labels, a, b, cell, y)
        for i in valid_idx:
            a, b, cell, y = samples[i]
            _append(valid_edges, valid_labels, a, b, cell, y)
        for i in test_idx:
            a, b, cell, y = samples[i]
            _append(test_edges, test_labels, a, b, cell, y)

        Synergistic_Graph, Antagonistic_Graph = [], []
        Synergistic_Types, Antagonistic_Types = [], []
        for c in all_cell_lines:
            cell_train = np.array(train_edges[c]).reshape(-1, 2) if len(train_edges[c]) else np.array([]).reshape(0, 2)
            if len(cell_train) > 0:
                cell_train = np.concatenate([cell_train, [[x[1], x[0]] for x in cell_train]])
            train_edges[c] = cell_train
            valid_edges[c] = np.array(valid_edges[c]).reshape(-1, 2) if len(valid_edges[c]) else np.array([]).reshape(0, 2)
            test_edges[c] = np.array(test_edges[c]).reshape(-1, 2) if len(test_edges[c]) else np.array([]).reshape(0, 2)

            y_cell_train = train_labels[c]
            pos_cnt = int(np.sum(np.array(y_cell_train) == 1)) if len(y_cell_train) else 0
            neg_cnt = int(np.sum(np.array(y_cell_train) == 0)) if len(y_cell_train) else 0
            pos_weights[c] = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0

            if len(train_labels[c]) > 0:
                orig_train = np.array(train_edges[c][:len(train_labels[c])]).reshape(-1, 2)
                for (a, b), y in zip(orig_train.tolist(), train_labels[c]):
                    if int(y) == 1:
                        Synergistic_Graph.append(sorted([int(a), int(b), int(c)]))
                        Synergistic_Types.append('synergistic')
                    else:
                        Antagonistic_Graph.append(sorted([int(a), int(b), int(c)]))
                        Antagonistic_Types.append('antagonistic')

        hyperedges = Synergistic_Graph + Antagonistic_Graph
        hyperedge_types = Synergistic_Types + Antagonistic_Types
        drug_disease_edges = []

    if drug_disease_pairs is not None and disease_offset is not None:
        for drug_id, disease_id in drug_disease_pairs:
            hyperedges.append([drug_id, disease_offset + disease_id])
            hyperedge_types.append('drug-disease')
        drug_disease_edges = [[int(d), int(disease_offset + dis)] for d, dis in drug_disease_pairs]
    if dti_edges is None:
        dti_edges = []

    if len(drug_disease_edges) > 0:
        node_num = disease_offset + num_disease + numProtein
    else:
        node_num = numDrug + numCellline + numProtein
        disease_offset = None

    hypergraph_groups = [Synergistic_Graph, Antagonistic_Graph, drug_disease_edges, dti_edges]
    hypergraph = Synergistic_Graph + Antagonistic_Graph + drug_disease_edges + dti_edges
    edge_length = [len(Synergistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph) + len(drug_disease_edges),
                   len(hypergraph)]
    V, E, degV = Construct_Hypergraph(hypergraph_groups, node_num)
    realFold = RealFoldData(train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights)
    realFold.iFold = iFold
    realFold.CellsCount = numCellline
    realFold.numDrug = numDrug
    realFold.numNode = node_num if len(drug_disease_edges) > 0 else (numDrug + numCellline)
    realFold.V = V
    realFold.E = E
    realFold.edge_length = edge_length
    realFold.degV_dict = degV
    realFold.hypergraph_edge_num = len(hypergraph)
    realFold.hyperedges = hyperedges
    realFold.hyperedge_types = hyperedge_types
    return realFold

 


def produce_drug_leaveout(data, drug_disease_pairs=None, disease_offset=None, dti_edges=None, numProtein=0, disease_features=None, num_disease=0):
    """按单药物 K 折留出测试；样本级分层切 valid；train 做对称增强。"""
    Interaction_Score, numDrug, numCellline, iFold = data
    train_edges = {}
    train_labels = {}
    test_edges = {}
    test_labels = {}
    valid_edges = {}
    valid_labels = {}
    pos_weights = {}
    Synergistic_Graph = []
    Antagonistic_Graph = []
    Synergistic_Types = []
    Antagonistic_Types = []

    Interaction_Score = Interaction_Score.copy()
    for ridx, row in Interaction_Score.iterrows():
        a, b = row['DrugA'], row['DrugB']
        Interaction_Score.at[ridx, 'DrugA'], Interaction_Score.at[ridx, 'DrugB'] = sorted([a, b])

    samples = []   # (drugA, drugB, cellLine, y)
    ys = []
    drug_ids = []  # (drugA, drugB) per sample
    for _, row in Interaction_Score.iterrows():
        score = row.iloc[3]
        if score >= params.threshold:
            y = 1
        elif score < 0:
            y = 0
        else:
            continue
        drugA = int(row['DrugA'])
        drugB = int(row['DrugB'])
        cell = int(row['CellLine'])
        samples.append((drugA, drugB, cell, y))
        ys.append(y)
        drug_ids.append((drugA, drugB))

    if len(samples) == 0:
        all_cell_lines = range(numDrug, numDrug + numCellline)
        for Cell_Line in all_cell_lines:
            train_edges[Cell_Line] = np.array([]).reshape(0, 2)
            train_labels[Cell_Line] = []
            valid_edges[Cell_Line] = np.array([]).reshape(0, 2)
            valid_labels[Cell_Line] = []
            test_edges[Cell_Line] = np.array([]).reshape(0, 2)
            test_labels[Cell_Line] = []
            pos_weights[Cell_Line] = 1.0
        hyperedges = []
        hyperedge_types = []
        drug_disease_edges = []
    else:
        from sklearn.model_selection import KFold, StratifiedShuffleSplit

        all_drugs = list(range(numDrug))
        kf = KFold(n_splits=params.k_fold, shuffle=True, random_state=params.seed)
        folds = list(kf.split(np.zeros(len(all_drugs))))
        if iFold < 0 or iFold >= len(folds):
            raise ValueError(f"Invalid fold index {iFold} for k_fold={params.k_fold}")

        train_drug_idx, test_drug_idx = folds[iFold]
        test_drug_set = {all_drugs[i] for i in test_drug_idx}

        test_idx = [
            i for i, (drugA, drugB, _, _) in enumerate(samples)
            if (drugA in test_drug_set) or (drugB in test_drug_set)
        ]
        trainval_idx = [
            i for i in range(len(samples)) if i not in test_idx
        ]

        valid_ratio = 1.0 / float(params.k_fold)
        valid_ratio = max(min(valid_ratio, 0.5), 0.05)

        y_trainval = np.array([ys[i] for i in trainval_idx]) if len(trainval_idx) else np.array([])
        if len(trainval_idx) == 0 or len(np.unique(y_trainval)) < 2:
            valid_idx = np.array([], dtype=int)
            train_idx = np.array(trainval_idx, dtype=int)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=params.seed + iFold)
            rel_train_idx, rel_valid_idx = next(sss.split(np.zeros(len(trainval_idx)), y_trainval))
            train_idx = np.array(trainval_idx, dtype=int)[rel_train_idx]
            valid_idx = np.array(trainval_idx, dtype=int)[rel_valid_idx]

        all_cell_lines = list(range(numDrug, numDrug + numCellline))
        for Cell_Line in all_cell_lines:
            train_edges[Cell_Line] = np.array([]).reshape(0, 2)
            train_labels[Cell_Line] = []
            valid_edges[Cell_Line] = np.array([]).reshape(0, 2)
            valid_labels[Cell_Line] = []
            test_edges[Cell_Line] = np.array([]).reshape(0, 2)
            test_labels[Cell_Line] = []
            pos_weights[Cell_Line] = 1.0

        def _append(split_edges, split_labels, drugA, drugB, cell, y):
            split_edges[cell].append([drugA, drugB])
            split_labels[cell].append(int(y))

        for Cell_Line in all_cell_lines:
            train_edges[Cell_Line] = []
            valid_edges[Cell_Line] = []
            test_edges[Cell_Line] = []

        for i in train_idx:
            drugA, drugB, cell, y = samples[i]
            _append(train_edges, train_labels, drugA, drugB, cell, y)
        for i in valid_idx:
            drugA, drugB, cell, y = samples[i]
            _append(valid_edges, valid_labels, drugA, drugB, cell, y)
        for i in test_idx:
            drugA, drugB, cell, y = samples[i]
            _append(test_edges, test_labels, drugA, drugB, cell, y)

        # --------- 5) 每个 cell：转 numpy、对称增强训练边、pos_weight、构建训练超边 ---------
        for Cell_Line in all_cell_lines:
            cell_train = np.array(train_edges[Cell_Line]).reshape(-1, 2) if len(train_edges[Cell_Line]) else np.array([]).reshape(0, 2)
            cell_valid = np.array(valid_edges[Cell_Line]).reshape(-1, 2) if len(valid_edges[Cell_Line]) else np.array([]).reshape(0, 2)
            cell_test = np.array(test_edges[Cell_Line]).reshape(-1, 2) if len(test_edges[Cell_Line]) else np.array([]).reshape(0, 2)

            if len(cell_train) > 0:
                cell_train = np.concatenate([cell_train, [[x[1], x[0]] for x in cell_train]])

            y_cell_train = train_labels[Cell_Line]
            pos_cnt = int(np.sum(np.array(y_cell_train) == 1)) if len(y_cell_train) else 0
            neg_cnt = int(np.sum(np.array(y_cell_train) == 0)) if len(y_cell_train) else 0
            pos_weight = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0

            train_edges[Cell_Line] = cell_train
            valid_edges[Cell_Line] = cell_valid
            test_edges[Cell_Line] = cell_test
            pos_weights[Cell_Line] = pos_weight

            if pos_cnt + neg_cnt > 0 and len(train_labels[Cell_Line]) > 0:
                orig_train = np.array(train_edges[Cell_Line][:len(train_labels[Cell_Line])]).reshape(-1, 2)
                for (a, b), y in zip(orig_train.tolist(), train_labels[Cell_Line]):
                    if int(y) == 1:
                        Synergistic_Graph.append(sorted([int(a), int(b), int(Cell_Line)]))
                        Synergistic_Types.append('synergistic')
                    else:
                        Antagonistic_Graph.append(sorted([int(a), int(b), int(Cell_Line)]))
                        Antagonistic_Types.append('antagonistic')

        hyperedges = Synergistic_Graph + Antagonistic_Graph
        hyperedge_types = Synergistic_Types + Antagonistic_Types
        drug_disease_edges = []

    # 其余部分与 produce_global 相同：添加疾病/蛋白质边并构建超图
    if drug_disease_pairs is not None and disease_offset is not None:
        for drug_id, disease_id in drug_disease_pairs:
            hyperedges.append([drug_id, disease_offset + disease_id])
            hyperedge_types.append('drug-disease')
        drug_disease_edges = [[int(d), int(disease_offset + dis)] for d, dis in drug_disease_pairs]
    if dti_edges is None:
        dti_edges = []

    if len(drug_disease_edges) > 0 and disease_offset is not None:
        protein_offset = disease_offset + num_disease
    else:
        protein_offset = numDrug + numCellline

    if len(drug_disease_edges) > 0:
        if disease_offset is not None:
            node_num = disease_offset + num_disease + numProtein
        else:
            max_dis = max([dis for _, dis in drug_disease_pairs]) if len(drug_disease_pairs) > 0 else -1
            numDisease = max_dis + 1
            disease_offset = numDrug + numCellline
            node_num = disease_offset + numDisease + numProtein
    else:
        node_num = numDrug + numCellline + numProtein
        disease_offset = None

    hypergraph_groups = [Synergistic_Graph, Antagonistic_Graph, drug_disease_edges, dti_edges]
    hypergraph = Synergistic_Graph + Antagonistic_Graph + drug_disease_edges + dti_edges
    edge_length = [len(Synergistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph) + len(drug_disease_edges),
                   len(hypergraph)]
    V, E, degV = Construct_Hypergraph(hypergraph_groups, node_num)
    realFold = RealFoldData(train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights)
    realFold.iFold = iFold
    realFold.CellsCount = numCellline
    realFold.numDrug = numDrug
    if len(drug_disease_edges) > 0:
        realFold.numNode = node_num
    else:
        realFold.numNode = numDrug + numCellline
    realFold.V = V
    realFold.E = E
    realFold.edge_length = edge_length
    realFold.degV_dict = degV
    realFold.hypergraph_edge_num = len(hypergraph)
    realFold.hyperedges = hyperedges
    realFold.hyperedge_types = hyperedge_types
    return realFold

def produce(data, drug_disease_pairs=None, disease_offset=None, dti_edges=None, numProtein=0, disease_features=None, num_disease=0):
    Interaction_Score, numDrug, numCellline, iFold = data
    train_edges = {}
    train_labels = {}
    test_edges = {}
    test_labels = {}
    valid_edges = {}
    valid_labels = {}
    pos_weights = {}
    Synergistic_Graph = []
    Antagonistic_Graph = []
    Synergistic_Types = []
    Antagonistic_Types = []
    Cell_Line_Specific = Interaction_Score.groupby('CellLine')
    for Cell_Line, Interaction in Cell_Line_Specific:
        for idx, (a, b) in Interaction[['DrugA', 'DrugB']].iterrows():
            Interaction.at[idx, 'DrugA'], Interaction.at[idx, 'DrugB'] = sorted([a, b])
        Positive_Interaction = Interaction[Interaction.iloc[:,3] >= params.threshold]
        Negitive_Interaction = Interaction[Interaction.iloc[:,3] < 0]
        Positive_Interaction = sorted(Positive_Interaction[['DrugA', 'DrugB']].values.tolist())
        Negitive_Interaction = sorted(Negitive_Interaction[['DrugA', 'DrugB']].values.tolist())
        random.seed(params.seed)
        random.shuffle(Positive_Interaction)
        random.shuffle(Negitive_Interaction)
        nSize = len(Positive_Interaction)
        foldSize = int(nSize / params.k_fold)
        startTest = iFold * foldSize
        endTest = (iFold + 1) * foldSize
        if endTest > nSize:
            endTest = nSize
        if iFold == params.k_fold - 1:
            startValid = 0
        else:
            startValid = endTest
        endValid = startValid + foldSize
        test_pos = Positive_Interaction[startTest:endTest]
        valid_pos = Positive_Interaction[startValid:endValid]
        test_neg = Negitive_Interaction[startTest:endTest]
        valid_neg = Negitive_Interaction[startValid:endValid]
        train_pos = [ x for x in Positive_Interaction if x not in test_pos + valid_pos ]
        train_neg = [ x for x in Negitive_Interaction if x not in test_neg + valid_neg ]
        Synergistic_Graph.extend([sorted(row + [Cell_Line]) for row in train_pos])
        Synergistic_Types.extend(['synergistic'] * len(train_pos))
        Antagonistic_Graph.extend([sorted(row + [Cell_Line]) for row in train_neg])
        Antagonistic_Types.extend(['antagonistic'] * len(train_neg))
        test = np.concatenate([test_pos, test_neg])
        y_test = [1] * len(test_pos) + [0] * len(test_neg)
        valid = np.concatenate([valid_pos, valid_neg])
        y_valid = [1] * len(valid_pos) + [0] * len(valid_neg)
        train = np.concatenate([train_pos, train_neg])
        y_train = [1] * len(train_pos) + [0] * len(train_neg)
        train = np.concatenate([train, [ [x[1],x[0]] for x in train ] ])
        train_edges[Cell_Line] = train
        train_labels[Cell_Line] = y_train
        test_edges[Cell_Line] = test
        test_labels[Cell_Line] = y_test
        valid_edges[Cell_Line] = valid
        valid_labels[Cell_Line] = y_valid
        pos_weight = len(train_neg) / len(train_pos)
        pos_weights[Cell_Line] = pos_weight
    # 合并超边和类型（仅用于保存原始列表）
    hyperedges = Synergistic_Graph + Antagonistic_Graph
    hyperedge_types = Synergistic_Types + Antagonistic_Types
    # 添加药物-疾病二元超边及类型（保存）
    if drug_disease_pairs is not None and disease_offset is not None:
        for drug_id, disease_id in drug_disease_pairs:
            hyperedges.append([drug_id, disease_offset + disease_id])
            hyperedge_types.append('drug-disease')
    # 三类超边用于构图：0=synergy, 1=antagonism, 2=drug-disease
    drug_disease_edges = []
    if drug_disease_pairs is not None and disease_offset is not None:
        drug_disease_edges = [[int(d), int(disease_offset + dis)] for d, dis in drug_disease_pairs]

    # 第四类：DTI 药物-蛋白质二元边
    if dti_edges is None:
        dti_edges = []

    # 节点总数：药物 + 细胞系 + 疾病(若有) + 蛋白质
    # 使用传入的蛋白质数量
    # 蛋白质节点应该排在疾病节点之后
    if len(drug_disease_edges) > 0 and disease_offset is not None:
        protein_offset = disease_offset + num_disease
    else:
        protein_offset = numDrug + numCellline
    
    if len(drug_disease_edges) > 0:
        # 使用传入的疾病数量，而不是从pairs中计算
        if disease_offset is not None:
            # 使用传入的疾病偏移量和数量
            node_num = disease_offset + num_disease + numProtein
        else:
            # 如果没有传入疾病信息，从pairs中计算
            max_dis = max([dis for _, dis in drug_disease_pairs]) if len(drug_disease_pairs) > 0 else -1
            numDisease = max_dis + 1
            disease_offset = numDrug + numCellline
            node_num = disease_offset + numDisease + numProtein
    else:
        node_num = numDrug + numCellline + numProtein
        disease_offset = None

    hypergraph_groups = [Synergistic_Graph, Antagonistic_Graph, drug_disease_edges, dti_edges]
    hypergraph = Synergistic_Graph + Antagonistic_Graph + drug_disease_edges + dti_edges
    edge_length = [len(Synergistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph),
                   len(Synergistic_Graph) + len(Antagonistic_Graph) + len(drug_disease_edges),
                   len(hypergraph)]
    V, E, degV = Construct_Hypergraph(hypergraph_groups, node_num)
    realFold = RealFoldData(train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights)
    realFold.iFold = iFold
    realFold.CellsCount = numCellline
    realFold.numDrug = numDrug
    # 修正：当存在疾病节点时，总节点数应包含疾病节点
    if len(drug_disease_edges) > 0:
        realFold.numNode = node_num
    else:
        realFold.numNode = numDrug + numCellline
    realFold.V = V
    realFold.E = E
    realFold.edge_length = edge_length
    realFold.degV_dict = degV
    realFold.hypergraph_edge_num = len(hypergraph)
    realFold.hyperedges = hyperedges
    realFold.hyperedge_types = hyperedge_types
    return realFold

class RealFoldData:
    def __init__(self, train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights):
        self.train_edges = train_edges
        self.train_labels = train_labels
        self.test_edges = test_edges
        self.test_labels = test_labels
        self.valid_edges = valid_edges
        self.valid_labels = valid_labels
        self.pos_weights = pos_weights
        self.hyperedges: list = []
        self.hyperedge_types: list = []
        self.iFold: int = 0
        self.CellsCount: int = 0
        self.numDrug: int = 0
        self.numNode: int = 0
        self.V = None
        self.E = None
        self.edge_length: list = []
        self.degV_dict: dict = {}
        self.hypergraph_edge_num: int = 0

def train_valid_test_split(Dataset_Name, DrugToID, numDrug):
    path = "%s/%s/" % (params.data_file, Dataset_Name)
    Interaction_Score = pd.read_csv(path + Dataset_Name + '_SCORE.csv',encoding='UTF-8')
    if params.Use_Cell_Line_Feature:
        Cell_Line = pd.read_csv(path + Dataset_Name + '_CELL_LINE_EXPRESSION.csv',encoding='UTF-8')
        CellLineToID = Cell_Line[['Cell_Line']].copy()
        CellLineToID['ID'] = range(numDrug, numDrug + len(CellLineToID))
    else:
        CellLineToID = Interaction_Score[['Cell_Line']].copy().drop_duplicates()
        CellLineToID['ID'] = range(numDrug, numDrug + len(CellLineToID))
    Cell_Line_Feature = Cell_Line.iloc[:,1:].to_numpy()
    # 静默
    Cell_Line_Feature = (Cell_Line_Feature - np.mean(Cell_Line_Feature, axis=0)) / np.std(Cell_Line_Feature, axis=0)
    Interaction_Score = (
        pd.merge(Interaction_Score, DrugToID.iloc[:,1:], left_on='Drug_A', right_on='PubChem_CID', how='inner')
        .merge(DrugToID.iloc[:,1:], left_on='Drug_B', right_on='PubChem_CID', how='inner')
        .merge(CellLineToID, left_on='Cell_Line', right_on='Cell_Line', how='inner')
    )
    Interaction_Score = Interaction_Score[['Drug_ID_x','Drug_ID_y','ID','Score']]
    Interaction_Score = Interaction_Score.rename(columns={'Drug_ID_x':'DrugA','Drug_ID_y':'DrugB','ID':'CellLine'})
    numCellline = len(CellLineToID)
    
    # 使用疾病关联数据
    drug_disease_pairs, num_disease, disease_offset, disease_features = load_disease1_data(DrugToID, numDrug, numCellline)

    # 基于 DTI 生成药物-药物对（共享至少一个靶点）
    try:
        # Drug_Target 在本模块全局不可见，这里从磁盘重建一次
        # 重用 process_data 的路径逻辑
        path = "%s/%s/" % (params.data_file, Dataset_Name)
        Drug_Information = pd.read_csv(path + Dataset_Name + '_DRUG.csv',encoding='UTF-8').drop_duplicates(subset = ['PubChem_CID'])
        Drug_Information['Drug_ID'] = range(len(Drug_Information))
        base_DrugToID = Drug_Information[['Name','PubChem_CID','Drug_ID']]
        Drug_Target_src = pd.read_csv(params.data_file +'/Chemical_Target_Interaction.csv',encoding='UTF-8')
        Drug_Target_src = pd.merge(Drug_Target_src, base_DrugToID.iloc[:,1:], left_on = 'PubChem_CID', right_on = 'PubChem_CID', how = 'inner')
        Target_List = Drug_Target_src['Entry_ID'].drop_duplicates().tolist()
        # 使用从0开始的连续索引，而不是基于base_DrugToID的长度
        Target = pd.Series(Target_List, index = range(len(Target_List)))
        TargetToID_tmp = pd.DataFrame({'Target': Target.values, 'Target_ID': Target.index})
        Drug_Target_src = pd.merge(Drug_Target_src, TargetToID_tmp, left_on = 'Entry_ID', right_on = 'Target', how = 'inner')
        Drug_Target_src = pd.DataFrame(Drug_Target_src[['Drug_ID','Target_ID']])
        # 构建药物-蛋白质二元边（DTI）
        dti_edges = []
        # 蛋白质节点应该排在疾病节点之后
        protein_offset = numDrug + numCellline + num_disease  # 计算蛋白质节点偏移量
        for _, row in Drug_Target_src.iterrows():
            drug_id = int(row['Drug_ID'])
            protein_id = int(row['Target_ID'])
            # 蛋白质节点ID需要加上偏移量
            protein_node_id = protein_offset + protein_id
            dti_edges.append([drug_id, protein_node_id])
        
        # 去重
        dti_edges = list(set(tuple(sorted(edge)) for edge in dti_edges))
        dti_edges = [list(edge) for edge in dti_edges]
    except Exception:
        dti_edges = None

    realFolds = {}
    split_id = _split_method_id(getattr(params, 'split_method', '1'))
    for iFold in range(params.k_fold):
        data = Interaction_Score, numDrug, numCellline, iFold
        if split_id == 2:
            fn = produce_cell_line_leaveout
        elif split_id == 3:
            fn = produce_drug_combo_leaveout
        else:
            fn = produce
        realFolds[iFold] = fn(
            data,
            drug_disease_pairs=drug_disease_pairs,
            disease_offset=disease_offset,
            dti_edges=dti_edges,
            numProtein=len(Target_List) if 'Target_List' in locals() else 0,
            disease_features=disease_features,
            num_disease=num_disease,
        )
    return realFolds, Cell_Line_Feature, CellLineToID, disease_features

def process_data(Dataset_Name):
    # 规范化数据集名：避免命令行/配置里误带空格/引号导致路径拼接出错
    # 例如：ONEIL 、"ONEIL"、'ONEIL'
    Dataset_Name = str(Dataset_Name).strip().strip('\'"')
    # Windows/Linux 兼容：用 os.path.join 拼路径，避免分隔符混用导致找不到文件
    import os
    # 允许忽略大小写/意外字符后匹配实际目录名（Windows 通常不区分大小写，但这里也做一层保护）
    data_root = params.data_file
    candidate = Dataset_Name
    try:
        subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for d in subdirs:
            if d.strip().strip('\'"').lower() == candidate.lower():
                candidate = d
                break
    except Exception:
        pass

    path = os.path.join(data_root, candidate)
    drug_path = os.path.join(path, f"{Dataset_Name}_DRUG.csv")
    if not os.path.exists(drug_path):
        # 再尝试一次：文件前缀也用 candidate（防止 Dataset_Name 与实际目录名大小写/格式不同）
        alt_drug_path = os.path.join(path, f"{candidate}_DRUG.csv")
        if os.path.exists(alt_drug_path):
            drug_path = alt_drug_path
        else:
            raise FileNotFoundError(
                f"未找到药物文件: {drug_path}。请检查 params.data_file 与 dataset 名称是否正确。"
            )
    Drug_Information = pd.read_csv(drug_path, encoding='UTF-8').drop_duplicates(subset=['PubChem_CID'])
    Drug_CID = Drug_Information['PubChem_CID'].tolist() # list
    Drug_Information['Drug_ID'] = range(len(Drug_Information))
    DrugToID = Drug_Information[['Name','PubChem_CID','Drug_ID']]
    
    # 提取SMILES信息（用于可选的分子图GCN，也用于传统指纹计算）
    smiles_list = Drug_Information['SMILES'].tolist()
    # 静默：避免运行时刷屏
    numDrug = len(DrugToID)
    # Drug feature matrix
    # 恢复原始设计：使用 Morgan 指纹 (radius=6, nBits=300) + 分子描述符
    Fingerprints = []
    descriptors = []
    nBits = 300
    for drug_id, smi in zip(Drug_Information['Drug_ID'], Drug_Information['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # 无效 SMILES：用全零向量代替
            fp_arr = np.zeros((nBits,), dtype=np.int32)
            descriptor = {}
        else:
            # Morgan 指纹（与项目文档一致：radius=6, nBits=300）
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=6, nBits=nBits)
            fp_arr = np.zeros((nBits,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            # 分子描述符
            descriptor = Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
        Fingerprints.append((str(drug_id), fp_arr.astype(np.float32).tolist()))
        descriptors.append((str(drug_id), descriptor))
    
    Fingerprints = pd.DataFrame(dict(Fingerprints)).transpose().to_numpy()
    descriptors = pd.DataFrame(dict(descriptors)).transpose().to_numpy()
    # 静默：避免运行时刷屏
    Drug_Features = np.hstack((Fingerprints, descriptors))
    # 静默：避免运行时刷屏
    Drug_Features = np.delete(Drug_Features, np.where(np.var(Drug_Features, axis=0) == 0)[0], axis=1)
    Drug_Features = Drug_Features[:, ~np.isnan(Drug_Features).any(axis=0)]
    # 静默：避免运行时刷屏
    Drug_Features = (Drug_Features - np.mean(Drug_Features, axis=0)) / np.std(Drug_Features, axis=0)

    realFolds, Cell_Line_Feature, CellLineToID, disease_features = train_valid_test_split(Dataset_Name, DrugToID, numDrug)

    # 返回训练所需的核心数据（已去掉未使用的 DDI / PPI / Drug_Target 矩阵）
    return (
        realFolds,
        Drug_Features,
        Cell_Line_Feature,
        DrugToID,
        CellLineToID,
        disease_features,
        smiles_list,
    )

def load_disease1_data(DrugToID, numDrug, numCellline):
    """
    加载疾病关联数据，包括药物-疾病关联和疾病嵌入特征
    
    Args:
        DrugToID: 药物ID映射表
        numDrug: 药物数量
        numCellline: 细胞系数量
    
    Returns:
        drug_disease_pairs: 药物-疾病关联对列表
        num_disease: 疾病数量
        disease_offset: 疾病节点偏移量
        disease_features: 疾病嵌入特征矩阵
    """
    try:
        # 读取药物-疾病关联数据
        drug_disease_df = pd.read_csv(os.path.join(params.data_file, 'DISEASE', 'drug_disease_indication.csv'))
        disease_info_df = pd.read_csv(os.path.join(params.data_file, 'DISEASE', 'DISEASE_INFO.csv'))
        
        # 读取疾病嵌入特征
        disease_embeddings = np.load(os.path.join(params.data_file, 'DISEASE', 'disease_embd.npy'))
        
        # 静默：避免运行时刷屏
        
        # 创建疾病ID到索引的映射
        disease_to_idx = {str(cui): i for i, cui in enumerate(disease_info_df['cuis'].astype(str).tolist())}
        # 静默
        
        # 收集实际使用的疾病CUI
        used_diseases = set()
        
        # 创建PubChem CID到Drug_ID的映射
        pubchem_to_id = {}
        if 'PubChem_CID' in DrugToID.columns:
            pubchem_to_id = {str(cid): int(did) for cid, did in zip(DrugToID['PubChem_CID'].astype(str).tolist(), DrugToID['Drug_ID'].tolist())}
        
        # 构建药物-疾病关联对
        drug_disease_pairs = []
        matched_count = 0
        
        for _, row in drug_disease_df.iterrows():
            pubchem_id = str(row['pubchemid'])
            cui = str(row['cuis'])
            
            # 检查疾病是否在疾病信息中
            if cui not in disease_to_idx:
                continue
            
            # 检查药物是否在主数据集中
            if pubchem_id in pubchem_to_id:
                drug_id = pubchem_to_id[pubchem_id]
                disease_idx = disease_to_idx[cui]
                drug_disease_pairs.append((drug_id, disease_idx))
                used_diseases.add(cui)
                matched_count += 1
        
        # 静默
        
        # 重新映射疾病ID，使其连续
        if len(used_diseases) > 0:
            # 创建新的连续疾病ID映射
            new_disease_to_idx = {cui: i for i, cui in enumerate(sorted(used_diseases))}
            
            # 重新构建药物-疾病对，使用新的连续ID
            drug_disease_pairs = []
            for _, row in drug_disease_df.iterrows():
                pubchem_id = str(row['pubchemid'])
                cui = str(row['cuis'])
                
                if cui in new_disease_to_idx and pubchem_id in pubchem_to_id:
                    drug_id = pubchem_to_id[pubchem_id]
                    disease_idx = new_disease_to_idx[cui]
                    drug_disease_pairs.append((drug_id, disease_idx))
            
            # 静默
        
        # 静默：不打印示例边
        
        # 计算疾病节点偏移量
        disease_offset = numDrug + numCellline
        # 使用实际使用的疾病数量
        num_disease = len(used_diseases) if len(used_diseases) > 0 else 0
        
        # 标准化疾病嵌入特征
        disease_features = disease_embeddings
        if len(disease_features.shape) == 1:
            disease_features = disease_features.reshape(1, -1)
        
        # 只取实际使用的疾病对应的特征
        if num_disease > 0 and len(used_diseases) > 0:
            # 获取实际使用的疾病在原特征矩阵中的索引
            used_indices = [disease_to_idx[cui] for cui in used_diseases]
            disease_features = disease_features[used_indices]
        
        # 标准化特征
        disease_features = (disease_features - np.mean(disease_features, axis=0)) / (np.std(disease_features, axis=0) + 1e-8)
        
        # 静默
        
        return drug_disease_pairs, num_disease, disease_offset, disease_features
        
    except Exception:
        return None, 0, None, None

def torch_from_numpy(Data, Device):
    # 仅遍历细胞系索引范围（避免将疾病节点当作细胞系）
    for idx in range(Data.numDrug, Data.numDrug + Data.CellsCount):
        Data.train_edges[idx] = torch.tensor(Data.train_edges[idx]).long().to(Device)
        Data.train_labels[idx] = torch.tensor(Data.train_labels[idx]).float().to(Device)
        Data.valid_edges[idx] = torch.tensor(Data.valid_edges[idx]).long().to(Device)
        Data.valid_labels[idx] = torch.tensor(Data.valid_labels[idx]).float().to(Device)
        Data.test_edges[idx] = torch.tensor(Data.test_edges[idx]).long().to(Device)
        Data.test_labels[idx] = torch.tensor(Data.test_labels[idx]).float().to(Device)
        Data.pos_weights[idx] = torch.tensor(Data.pos_weights[idx]).float().to(Device)
    for i in range(len(Data.degV_dict)):
        Data.degV_dict[i] = torch.tensor(Data.degV_dict[i]).float().to(Device)
    Data.V = torch.tensor(Data.V).long().to(Device)
    Data.E = torch.tensor(Data.E).long().to(Device)
    return Data

 
