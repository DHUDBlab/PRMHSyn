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
import numpy as np


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
