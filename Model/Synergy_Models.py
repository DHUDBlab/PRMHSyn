import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from torch_scatter import scatter
# from torch_geometric.utils import softmax
from molecular_gcn_encoder import MolecularGCNEncoder, HighwayDrugFeatureFusion

############################################
#                BioEncoder
############################################
class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, disease_dim, num_cellline, num_protein, embeddingSize, device, protein_features=None, smiles_list=None, use_molecular_gcn=True, use_layer_norm=True):
        super(BioEncoder, self).__init__()
        self.device = device
        self.use_molecular_gcn = use_molecular_gcn
        self.use_layer_norm = use_layer_norm
        
        # -------drug_layer-------
        if self.use_molecular_gcn and smiles_list is not None:
            # 使用分子图GCN + 传统特征 Highway 融合
            self.molecular_gcn = MolecularGCNEncoder(
                hidden_dim=128,
                output_dim=256,
                num_layers=3,
                dropout=0.1,
                device=device
            ).to(device)
            
            # GCN 输出 + 传统 Drug_Features 融合到 embeddingSize
            self.drug_fusion = HighwayDrugFeatureFusion(
                gcn_dim=256,
                traditional_dim=dim_drug,
                output_dim=embeddingSize,
                num_highway_layers=2  # 可调
            ).to(device)
            
            # LayerNorm for drug output
            if self.use_layer_norm:
                self.drug_norm = nn.LayerNorm(embeddingSize)
            else:
                self.drug_norm = None
            
            # 保存SMILES列表
            self.smiles_list = smiles_list
            # 静默
        else:
            # 使用传统方法：MLP降维
            # MLP: dim_drug → hidden_dim → embeddingSize
            hidden_dim = 1024  # 中间层维度
            self.drug_mlp = nn.Sequential(
                nn.Linear(dim_drug, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embeddingSize)
            )
            # 初始化权重
            for m in self.drug_mlp.modules():
                if isinstance(m, nn.Linear):
                    self._init_weights(m)
            if self.use_layer_norm:
                self.drug_norm = nn.LayerNorm(embeddingSize)
            else:
                self.drug_norm = None
            # 静默

        
        # -------cell line_layer-------
        self.cell1 = nn.Linear(dim_cellline, dim_cellline//2)
        self._init_weights(self.cell1)
        if self.use_layer_norm:
            self.cell1_norm = nn.LayerNorm(dim_cellline//2)
        else:
            self.cell1_norm = None
        self.cell2 = nn.Linear(dim_cellline//2, embeddingSize)
        self._init_weights(self.cell2)
        if self.use_layer_norm:
            self.cell2_norm = nn.LayerNorm(embeddingSize)
        else:
            self.cell2_norm = None

        # -------disease_layer-------
        self.disease = nn.Linear(disease_dim, embeddingSize)
        self._init_weights(self.disease)
        if self.use_layer_norm:
            self.disease_norm = nn.LayerNorm(embeddingSize)
        else:
            self.disease_norm = None

        # -------protein_layer-------
        if protein_features is not None:
            # 使用预计算的蛋白质文本特征
            self.protein_features = torch.tensor(protein_features).float().to(device)
            self.protein = nn.Linear(protein_features.shape[1], embeddingSize)
            self._init_weights(self.protein)
            if self.use_layer_norm:
                self.protein_norm = nn.LayerNorm(embeddingSize)
            else:
                self.protein_norm = None
            self.use_text_features = True
        else:
            # 使用可学习嵌入（原始方法）
            self.protein = nn.Embedding(num_protein, embeddingSize)
            nn.init.kaiming_uniform_(self.protein.weight.data, nonlinearity='relu')
            proteinIndices = [i for i in range(num_protein)]
            self.proteinIndices = torch.from_numpy(np.asarray(proteinIndices)).long().to(self.device)
            self.use_text_features = False
        
        
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0.0)

    def forward(self, Drug_Features, Cell_Line_Feature, Disease_Feature):
        # -------drug_layer-------
        if self.use_molecular_gcn and hasattr(self, 'molecular_gcn'):
            # 使用分子图编码器 + 传统特征 Highway 融合
            gcn_features = self.molecular_gcn(self.smiles_list).to(self.device)
            Drug_Features = Drug_Features.to(self.device)
            x_drug = self.drug_fusion(gcn_features, Drug_Features)
            if self.drug_norm is not None:
                x_drug = self.drug_norm(x_drug)
            x_drug = torch.relu(x_drug)
        else:
            # 使用传统方法：MLP降维
            x_drug = self.drug_mlp(Drug_Features)
            if self.drug_norm is not None:
                x_drug = self.drug_norm(x_drug)
            x_drug = torch.relu(x_drug)

        # -------cell line_layer-------
        x_cell = self.cell1(Cell_Line_Feature)
        if self.cell1_norm is not None:
            x_cell = self.cell1_norm(x_cell)
        x_cell = torch.relu(x_cell)
        x_cell = self.cell2(x_cell)
        if self.cell2_norm is not None:
            x_cell = self.cell2_norm(x_cell)
        x_cell = torch.relu(x_cell)

        # -------disease_layer-------
        x_disease = self.disease(Disease_Feature)
        if self.disease_norm is not None:
            x_disease = self.disease_norm(x_disease)
        x_disease = torch.relu(x_disease)

        # -------protein_layer-------
        if self.use_text_features:
            x_protein = self.protein(self.protein_features)
            if self.protein_norm is not None:
                x_protein = self.protein_norm(x_protein)
            x_protein = torch.relu(x_protein)
        else:
            x_protein = self.protein(self.proteinIndices)
        
        return x_drug, x_cell, x_disease, x_protein



############################################
#          SE Layer (kept, used)
############################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x.shape: [batch_size, channel]
        y = self.fc(x)  # y.shape: [batch_size, channel]
        return x * y


############################################
#        PageRank (Sparse Version)
############################################
def pagerank_score_sparse(num_nodes, vertex, edges, num_iter=20, alpha=0.85, device=None):
    # edges, vertex 是 torch tensors on device 或 numpy arrays
    # 构造 incidence 矩阵 B (N x E) 的稀疏表示（使用 scipy）
    if isinstance(edges, torch.Tensor):
        edges_np = edges.cpu().numpy()
    else:
        edges_np = np.asarray(edges)
    if isinstance(vertex, torch.Tensor):
        vertex_np = vertex.cpu().numpy()
    else:
        vertex_np = np.asarray(vertex)

    E = int(edges_np.max()) + 1
    rows = vertex_np
    cols = edges_np
    data = np.ones_like(rows, dtype=np.float32)
    B = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, E)).tocsr()

    # adj_node = B * B.T  (稀疏)
    adj_node = B.dot(B.T).tocsr()
    adj_node.setdiag(0)  # 去掉自环

    # 归一化按行
    row_sum = np.array(adj_node.sum(axis=1)).squeeze()
    row_sum[row_sum == 0] = 1.0
    inv = 1.0 / row_sum
    D_inv = sp.diags(inv)

    M = D_inv.dot(adj_node)  # row-normalized sparse matrix

    pr = np.ones((num_nodes, 1), dtype=np.float32) / num_nodes
    for _ in range(num_iter):
        pr = alpha * (M.T.dot(pr)) + (1 - alpha) / num_nodes

    pr_torch = torch.from_numpy(pr).to(device) if device is not None else torch.from_numpy(pr)
    return pr_torch  # [N,1]


############################################
#           rhgcnConv Layer
############################################
class rhgcnConv(nn.Module):
    def __init__(self, edge_num, Xe_class_length, degV_dict, in_channels, out_channels, num_edge_types=3, negative_slope=0.2, use_norm=True, use_pagerank=True):
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=True) for _ in range(num_edge_types + 1)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.num_edge_types = num_edge_types
        self.reset_parameters()

        self.edge_num = edge_num
        self.Xe_class_length = Xe_class_length
        self.use_norm = use_norm
        self.use_pagerank = use_pagerank
        self.degV_dict = degV_dict
        # The hypergraph is fixed for one model instance, so its PageRank is too.
        self._pagerank = None

    def reset_parameters(self):
        for layer in self.W:
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)


    def forward(self, X, vertex, edges):
        X0 = self.W[0](X)
        if self.use_pagerank:
            if (self._pagerank is None or
                    self._pagerank.shape[0] != X.shape[0] or
                    self._pagerank.device != X.device):
                self._pagerank = pagerank_score_sparse(
                    X.shape[0],
                    vertex.cpu().numpy(),
                    edges.cpu().numpy(),
                    device=X.device
                )
            pr = self._pagerank.to(dtype=X.dtype)
        else:
            pr = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)


        # 节点→超边（加PageRank权重）
        Xve = []
        for i in range(self.num_edge_types):
            # 节点→超边：用PageRank加权节点特征
            node_feat = self.W[i + 1](X) * pr  # PR(j) * x_j
            node_feat = node_feat * self.degV_dict[i]
            node_feat = node_feat[vertex]
            Xve.append(node_feat)

        Xe1 = []
        for i in range(self.num_edge_types):
            # 超边特征聚合：平均聚合
            Xe1.append(scatter(Xve[i], edges, dim=0, reduce='mean', dim_size=self.edge_num))

        Xe = Xe1[0][:self.Xe_class_length[0], :]
        for i in range(self.num_edge_types - 1):
            Xe = torch.cat((Xe, Xe1[i + 1][self.Xe_class_length[i]:self.Xe_class_length[i + 1], :]), 0)

        # 超边→节点：不使用PageRank加权超边特征
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=X.shape[0])
        Xv = Xv + X0

        if self.use_norm:
            Xv = F.normalize(Xv, p=2, dim=1)  # L2 归一化

        Xv = self.leaky_relu(Xv)
        return Xv


############################################
#                 RHGNN
############################################
class RHGNN(nn.Module):
    def __init__(self, V, E, edge_num, Xe_class_length, degV_dict, nfeat, nhid, out_dim, num_edge_types, dropout, hyperedges, hyperedge_types, use_layer_norm=True, use_pagerank=True, use_se=True):

        super().__init__()
        self.conv_in = rhgcnConv(edge_num, Xe_class_length, degV_dict, nfeat, nhid, num_edge_types, use_pagerank=use_pagerank)
        self.se_in = SELayer(nhid) if use_se else nn.Identity()
        if use_layer_norm:
            self.norm_in = nn.LayerNorm(nhid)  # LayerNorm after first conv
        else:
            self.norm_in = None

        self.conv_out1 = rhgcnConv(edge_num, Xe_class_length, degV_dict, nhid, out_dim, num_edge_types, use_pagerank=use_pagerank)
        self.se_out1 = SELayer(out_dim) if use_se else nn.Identity()
        if use_layer_norm:
            self.norm_out = nn.LayerNorm(out_dim)  # LayerNorm after second conv
        else:
            self.norm_out = None

        self.V = V
        self.E = E
        self.dropout = nn.Dropout(dropout)
        self.hyperedges = hyperedges
        self.hyperedge_types = hyperedge_types

    def forward(self, X):

        X = self.conv_in(X, self.V, self.E)
        if self.norm_in is not None:
            X = self.norm_in(X)  # LayerNorm
        X = self.se_in(X)        # SE模块作用
        X = self.dropout(X)

        X = self.conv_out1(X, self.V, self.E)
        if self.norm_out is not None:
            X = self.norm_out(X)  # LayerNorm
        X = self.se_out1(X)      # SE模块作用

        return X


############################################
#       CrossGatedFiLM Synergy Decoder
############################################
class CrossGatedFiLM_SynergyDecoder(nn.Module):

    def __init__(self, emb_dim=300, hidden_dim=256, output_dim=1, dropout=0.1, numDrug=38, use_layer_norm=True):
        super().__init__()
        
        self.numDrug = numDrug
        self.emb_dim = emb_dim

        # ---------- FiLM γ-only ----------
        self.film_gamma = nn.Linear(emb_dim, emb_dim)

        # ---------- Cross-Gating ----------
        self.gate_d1_from_d2 = nn.Linear(emb_dim, emb_dim)
        self.gate_d2_from_d1 = nn.Linear(emb_dim, emb_dim)

        # ---------- Fusion ----------
        self.fusion_layer = nn.Linear(emb_dim * 2, hidden_dim)
        if use_layer_norm:
            self.fusion_norm = nn.LayerNorm(hidden_dim)  # LayerNorm after fusion
        else:
            self.fusion_norm = None

        # ---------- MLP ----------
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_embedding, combination_index, cell_embeddings):
        outputs = {}
        for cellidx in sorted(combination_index.keys()):
            pairs = combination_index[cellidx]
            local_idx = cellidx - self.numDrug
            assert 0 <= local_idx < cell_embeddings.shape[0]
            cell_emb = cell_embeddings[local_idx]

            d1 = drug_embedding[pairs[:, 0]]
            d2 = drug_embedding[pairs[:, 1]]

            c = cell_emb.unsqueeze(0).expand(d1.shape[0], -1)
            gamma = torch.tanh(self.film_gamma(c))
            d1_c = d1 * (1 + gamma)
            d2_c = d2 * (1 + gamma)

            g1 = torch.sigmoid(self.gate_d1_from_d2(d2_c))
            g2 = torch.sigmoid(self.gate_d2_from_d1(d1_c))
            d1_g = d1_c * (0.5 + g1)
            d2_g = d2_c * (0.5 + g2)

            fused = torch.cat([d1_g, d2_g], dim=1)
            fused = self.fusion_layer(fused)
            if self.fusion_norm is not None:
                fused = self.fusion_norm(fused)
            outputs[cellidx] = self.predictor(fused).squeeze(-1)

        return outputs

############################################
#                  Synergy Model
############################################
class PRMHSyn(nn.Module):
    def __init__(self, numDrug, BioEncoder, encoder2, decoder):
        super(PRMHSyn, self).__init__()
        self.BioEncoder = BioEncoder
        self.hgnn_encoder2 = encoder2
        self.decoder = decoder
        self.numDrug = numDrug
        
    def forward(self, Drug_Features, Cell_Line_Feature, Disease_Feature, combination_index):
        # Step 1: 编码 drug, cell line, disease, protein
        x_drug, x_cell, x_disease, x_protein = self.BioEncoder(
            Drug_Features, Cell_Line_Feature, Disease_Feature
        )

        # Step 2: 构建超图输入
        hypergraph2 = torch.cat((x_drug, x_cell, x_disease, x_protein), 0)

        # Step 3: 超图编码
        embedding2 = self.hgnn_encoder2(hypergraph2)

        # Step 4: 获取各类节点 embedding
        drug_embedding = embedding2[:self.numDrug, :]
        cell_embeddings = embedding2[self.numDrug:self.numDrug + x_cell.shape[0], :]

        # Step 5: ★★★ 关键修改：调用新的 CrossGatedFiLM Synergy Decoder ★★★
        result = self.decoder(drug_embedding, combination_index, cell_embeddings)

        # Step 6: 返回值（保持你的格式不变）
        return (
            result,
            torch.zeros(0, embedding2.shape[1], device=embedding2.device, dtype=embedding2.dtype),
            embedding2[self.numDrug:, :],
            drug_embedding,
            cell_embeddings
        )
