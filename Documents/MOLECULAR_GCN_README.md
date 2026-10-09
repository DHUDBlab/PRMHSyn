# 分子图GCN编码器使用说明

## 概述

我已经成功为你的PRMHSyn项目实现了分子图GCN编码器，它可以将药物SMILES转换为分子图，使用GCN进行编码，并通过注意力机制与现有药物特征融合。

## 实现的功能

### 1. 分子图GCN编码器 (`molecular_gcn_encoder.py`)

**MolecularGCNEncoder类**：
- 将SMILES字符串转换为分子图
- 使用GCN层进行图卷积
- 全局最大池化得到分子表示
- 支持批处理

**DrugFeatureFusion类**：
- 使用多头注意力机制融合GCN特征和传统特征
- 输出统一的药物表示

### 2. 修改的文件

**Synergy_Models.py**：
- 修改了`BioEncoder`类，添加了分子图GCN编码器支持
- 新增参数：`smiles_list`, `use_molecular_gcn`
- 在forward方法中集成了GCN编码和特征融合

**Data_Process.py**：
- 在`process_data`函数中提取SMILES信息
- 修改返回值，包含`smiles_list`

**Model/PRMHSyn.py**：
- 修改了BioEncoder的实例化，传入SMILES列表
- 启用分子图GCN编码器

## 使用方法

### 1. 安装依赖

```bash
pip install torch-geometric>=2.0.0
pip install rdkit>=2022.0.0
pip install torch-scatter>=2.0.0
pip install torch-sparse>=0.6.0
```

### 2. 运行测试

```bash
python test_molecular_gcn.py
```

### 3. 训练模型

```bash
python Model/PRMHSyn.py
```

## 技术细节

### 原子特征 (78维)
- 原子类型 (10维 one-hot)
- 度 (7维 one-hot)
- 形式电荷 (7维 one-hot)
- 杂化类型 (5维 one-hot)
- 芳香性 (1维)
- 氢原子数量 (5维 one-hot)
- 手性中心 (1维)
- 原子质量 (1维，归一化)
- 电负性 (1维，归一化)

### 键特征 (4维)
- 键类型 (4维 one-hot)
- 共轭 (1维)
- 环中 (1维)
- 立体化学 (1维)

### GCN架构
- 输入维度：78 (原子特征)
- 隐藏维度：128
- 输出维度：256
- 层数：3层
- Dropout：0.1

### 注意力融合
- 多头注意力：8个头
- 融合维度：512
- 使用自注意力机制

## 优势

1. **更丰富的分子表示**：GCN能够捕获分子的拓扑结构和化学键信息
2. **注意力融合**：自动学习GCN特征和传统特征的重要性权重
3. **端到端训练**：整个编码过程可以端到端训练
4. **鲁棒性**：对无效SMILES有容错处理
5. **可扩展性**：可以轻松调整GCN层数和特征维度

## 性能考虑

- GCN编码比传统特征提取稍慢，但能提供更丰富的表示
- 支持批处理，提高效率
- 内存使用合理，适合大规模数据集

## 配置选项

在`BioEncoder`初始化时可以调整：
- `use_molecular_gcn=True/False`：启用/禁用分子图GCN
- GCN参数：层数、隐藏维度、输出维度等
- 注意力参数：头数、融合维度等

## 注意事项

1. 确保安装了所有必要的依赖包
2. 如果遇到CUDA相关错误，可以设置`use_molecular_gcn=False`回退到传统方法
3. 首次运行时会比较慢，因为需要构建分子图
4. 建议在GPU上运行以获得更好的性能

## 未来改进

1. 可以尝试更复杂的图神经网络架构（如GraphSAGE、GAT等）
2. 可以添加更多的分子特征（如3D结构信息）
3. 可以优化注意力机制的设计
4. 可以添加预训练的分子表示模型

这个实现为你的药物协同预测任务提供了更强大的分子表示能力，应该能够提升模型的性能！
