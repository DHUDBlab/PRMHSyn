# -*- coding: utf-8 -*-
"""
OGB 64维原子特征 + GCNConv 分子图编码器
适配 PRMHSyn 的 BioEncoder，可直接替换旧版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import numpy as np
from torch_scatter import scatter_add


# ================================================================
#  OGB Official 64-dimensional Atomic Features
# ================================================================
allowable_atom_list = [
    'H','C','N','O','F','P','S','Cl','Br','I',
    'Si','B','Se','Zn','Cu','Na','Mg','K','Ca',
    'Fe','As','Al','Mn','Hg','Pb','Sn','Co','Cr','Ag','Ni',
    'Li','Bi','V','Sb','Ge','Zr','Mo','Ru','Rh','Pd','Cd','Pt','Au','Tl'
]

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[0]
    return [int(x == a) for a in allowable_set]


def ogb_atom_features(atom):
    """Construct 64-dim OGB atom feature."""
    features = []

    # 1) Atom type (44-dim)
    features += one_hot(atom.GetSymbol(), allowable_atom_list)

    # 2) Degree (0–5)
    features += one_hot(atom.GetDegree(), list(range(6)))

    # 3) Formal charge (1-dim)
    features.append(atom.GetFormalCharge())

    # 4) Chirality tag (4-dim)
    chiral_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    features += one_hot(atom.GetChiralTag(), chiral_list)

    # 5) Is chiral center (1)
    features.append(int(atom.HasProp('_ChiralityPossible')))

    # 6) Is in ring (1)
    features.append(int(atom.IsInRing()))

    # 7) Aromatic (1)
    features.append(int(atom.GetIsAromatic()))

    # 8) Hybridization (5-dim)
    hybrid_list = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2
    ]
    features += one_hot(atom.GetHybridization(), hybrid_list)

    # 9) Atomic mass (1)
    features.append(atom.GetMass() / 200.0)

    # 必须是 64 维
    assert len(features) == 64, f"Atomic feature dim must be 64, got {len(features)}"

    return features



# ================================================================
#  GCN Encoder with 64-dim OGB atom features (2D 分子图)
# ================================================================
class MolecularGCNEncoder(nn.Module):

    def __init__(self, hidden_dim=128, output_dim=256, num_layers=3, dropout=0.1, device=None):
        super().__init__()

        self.in_dim = 64     # 输入维度固定为 64
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = torch.device("cpu") if device is None else device

        # GCN层
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(self.in_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layers.append(GCNConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        node_features = [ogb_atom_features(atom) for atom in mol.GetAtoms()]

        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])

        if len(edges) == 0:
            edges = [[0, 0]]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def forward(self, smiles_list):
        data_list, valid_idx = [], []

        for i, smi in enumerate(smiles_list):
            g = self.smiles_to_graph(smi)
            if g is not None:
                data_list.append(g)
                valid_idx.append(i)

        if len(data_list) == 0:
            return torch.zeros(len(smiles_list), self.output_dim, device=self.device)

        batch = Batch.from_data_list(data_list).to(self.device)
        x, edge_index = batch.x, batch.edge_index

        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        mol_emb = global_max_pool(x, batch.batch)

        if len(valid_idx) < len(smiles_list):
            full = torch.zeros(len(smiles_list), self.output_dim, device=self.device)
            full[valid_idx] = mol_emb
            return full

        return mol_emb



class Highway(nn.Module):
    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = F.relu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear
            x = self.dropout(x)
        return x


class HighwayDrugFeatureFusion(nn.Module):
    def __init__(self, gcn_dim=256, traditional_dim=1024, output_dim=300, num_highway_layers=2):
        super().__init__()

        self.input_dim = gcn_dim + traditional_dim
        self.output_dim = output_dim

        # Highway Network
        self.highway = Highway(num_highway_layers, self.input_dim)

        # 映射到最终 embedding
        self.projection = nn.Linear(self.input_dim, output_dim)
        nn.init.kaiming_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, gcn_features, traditional_features):
        # 拼接
        x = torch.cat([gcn_features, traditional_features], dim=1)

        # Highway network 融合
        x = self.highway(x)

        # 映射到最终药物 embedding
        x = self.projection(x)

        return x


################################################################
#   EGNN Layer (简化版) & 3D 等变分子编码器
################################################################
class EGNNLayer(nn.Module):
    """
    简化版 EGNN layer：对节点特征 x 和坐标 pos 做等变更新。
    参考：E(n) Equivariant Graph Neural Networks.
    """
    def __init__(self, in_dim):
        super().__init__()
        hidden = in_dim
        # edge/update functions
        self.phi_e = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(in_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x, pos, edge_index):
        """
        x:   [N, F]
        pos: [N, 3]
        edge_index: [2, E]
        """
        row, col = edge_index  # i←j
        xi, xj = x[row], x[col]
        diff = pos[row] - pos[col]             # [E, 3]
        dist2 = (diff ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        m_ij = self.phi_e(torch.cat([xi, xj, dist2], dim=-1))  # [E, H]
        # 坐标更新：等变
        delta_pos = diff * self.phi_x(m_ij)  # [E, 3]
        pos = pos + scatter_add(delta_pos, row, dim=0, dim_size=x.size(0))

        # 特征更新
        m_i = scatter_add(m_ij, row, dim=0, dim_size=x.size(0))  # [N, H]
        x = self.phi_h(torch.cat([x, m_i], dim=-1))
        return x, pos


class EquivariantMolecularEncoder(nn.Module):
    """
    使用 3D 坐标 + EGNN 的分子编码器。
    - 对 SMILES 生成 3D 构象（RDKit ETKDG + 能量优化）
    - 使用简化 EGNN 更新原子特征与坐标
    - global pooling 得到每个分子的 embedding
    """
    def __init__(self, hidden_dim=128, output_dim=256, num_layers=3, dropout=0.1, device=None):
        super().__init__()
        self.in_dim = 64       # 与 OGB 原子特征保持一致
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = torch.device("cpu") if device is None else device

        self.input_proj = nn.Linear(self.in_dim, hidden_dim)
        self.egnn_layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def smiles_to_graph_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 添加氢原子并生成 3D 构象
        mol = Chem.AddHs(mol)
        try:
            # 关闭 RDKit 日志，避免打印 UFFTYPE 警告
            lg = RDLogger.logger()
            old_level = lg.level
            lg.setLevel(RDLogger.CRITICAL)

            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                # 恢复日志级别后返回
                lg.setLevel(old_level)
                return None
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)

            # 恢复原来的日志级别
            lg.setLevel(old_level)
        except Exception:
            return None

        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        node_features = [ogb_atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
        coords = []
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])

        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        if len(edges) == 0:
            edges = [[0, 0]]

        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(coords, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, pos=pos)

    def forward(self, smiles_list):
        data_list, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            g = self.smiles_to_graph_3d(smi)
            if g is not None:
                data_list.append(g)
                valid_idx.append(i)

        if len(data_list) == 0:
            return torch.zeros(len(smiles_list), self.output_dim, device=self.device)

        batch = Batch.from_data_list(data_list).to(self.device)
        x, edge_index, pos, batch_idx = batch.x, batch.edge_index, batch.pos, batch.batch

        x = self.input_proj(x)
        x = self.act(x)

        for layer in self.egnn_layers:
            x, pos = layer(x, pos, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        # 池化得到分子级 embedding
        mol_emb = global_max_pool(x, batch_idx)
        mol_emb = self.out_proj(mol_emb)

        if len(valid_idx) < len(smiles_list):
            full = torch.zeros(len(smiles_list), self.output_dim, device=self.device)
            full[valid_idx] = mol_emb
            return full

        return mol_emb



def test_molecular_gcn():
    """测试分子图GCN编码器"""
    # 测试SMILES
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "C1=CC=CC=C1",  # 苯
        "invalid_smiles",  # 无效SMILES
        "CCN(CC)CC"  # 三乙胺
    ]
    
    # 创建编码器
    encoder = MolecularGCNEncoder()
    
    # 测试编码
    features = encoder(test_smiles)
    print(f"输入SMILES数量: {len(test_smiles)}")
    print(f"输出特征形状: {features.shape}")
    print(f"特征示例: {features[0][:10]}")


if __name__ == "__main__":
    test_molecular_gcn()
