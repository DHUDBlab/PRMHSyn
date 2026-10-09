# 新药物-疾病数据集集成指南

## 数据格式要求

### 1. 输入数据格式
您的药物-疾病关联数据应该包含以下列：

```csv
drug_name,disease_name,drug_id,disease_id,evidence_type
Fluorouracil,Colorectal Cancer,3385,MESH:D003110,therapeutic
Cisplatin,Lung Cancer,441239,MESH:D008175,therapeutic
```

### 2. 必需字段
- **drug_name**: 药物名称（用于与ALMANAC匹配）
- **disease_name**: 疾病名称
- **drug_id**: 药物ID（可选，用于备用匹配）
- **disease_id**: 疾病ID（建议使用MeSH ID格式）
- **evidence_type**: 证据类型（可选）

### 3. 可选字段
- **pubchem_cid**: PubChem CID（用于精确匹配）
- **confidence_score**: 置信度分数
- **source**: 数据来源

## 集成步骤

### 步骤1: 准备数据文件
将您的数据保存为CSV格式，放在项目根目录下。

### 步骤2: 修改集成脚本
编辑 `integrate_new_drug_disease_dataset.py` 文件：

1. 修改 `new_data_path` 为您的数据文件路径
2. 根据您的列名修改数据读取逻辑
3. 调整药物名称匹配规则

### 步骤3: 运行集成脚本
```bash
python integrate_new_drug_disease_dataset.py
```

### 步骤4: 更新Data_Process.py
将集成结果添加到 `Data_Process.py` 中：

```python
# 在适当位置添加
drug_disease_pairs = load_new_drug_disease_data('your_data.csv')
```

## 数据匹配策略

### 药物匹配
1. **名称匹配**: 使用药物名称进行模糊匹配
2. **ID匹配**: 使用PubChem CID进行精确匹配
3. **别名匹配**: 考虑药物的别名和商品名

### 疾病匹配
1. **MeSH ID**: 优先使用MeSH ID
2. **疾病名称**: 使用标准化的疾病名称
3. **同义词**: 考虑疾病的同义词

## 注意事项

1. **数据质量**: 确保药物和疾病名称的标准化
2. **ID格式**: 建议使用标准的ID格式（如MeSH ID）
3. **重复处理**: 处理重复的药物-疾病关联
4. **缺失值**: 处理缺失的药物或疾病信息

## 示例数据

```csv
drug_name,disease_name,drug_id,disease_id,evidence_type,pubchem_cid
Fluorouracil,Colorectal Neoplasms,3385,MESH:D003110,therapeutic,3385
Cisplatin,Lung Neoplasms,441239,MESH:D008175,therapeutic,441239
Doxorubicin,Breast Neoplasms,31703,MESH:D001943,therapeutic,31703
Paclitaxel,Ovarian Neoplasms,36314,MESH:D010051,therapeutic,36314
Tamoxifen,Breast Neoplasms,2733526,MESH:D001943,therapeutic,2733526
```

## 支持的数据源

- DrugBank
- ChEMBL
- CTD (Comparative Toxicogenomics Database)
- DisGeNET
- SIDER
- 自定义数据集
