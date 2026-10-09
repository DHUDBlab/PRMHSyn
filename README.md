# PRMHSyn

We introduce **PRMHSyn**, a PageRank-enhanced relational multimodal hypergraph learning framework for predicting anti-cancer drug synergy.

## Requirements

- Python 3.9.22
- NumPy 1.26.4
- pandas 2.2.3
- RDKit 2025.03.2
- scikit-learn 1.6.1
- SciPy 1.13.1
- PyTorch 2.0.1+cu118
- CUDA 11.8
- PyTorch Geometric 2.5.3
- torch-scatter 2.1.2+pt20cu118
- torch-sparse 0.6.18+pt20cu118

## Installation

Activate the Conda environment used for the experiments:

```bash
conda activate my_rdkit_env
```

## Evaluation Settings

| Setting | `--split_method` | Description |
|---|---|---|
| Cell-line-wise split | `cell_line` | Splits samples independently within each cell line. |
| Cell-line leave-out | `cell_line_leaveout` | Evaluates generalization to unseen cell lines. |
| Drug-combination leave-out | `drug_combo_leaveout` | Evaluates generalization to unseen drug combinations. |

## Running the Model

Run PRMHSyn on ONEIL:

```bash
python Model/PRMHSyn.py --dataset ONEIL --threshold 30 --split_method cell_line --learning_rate 5e-4 --weight_decay 5e-5 --epochs 2000
```

Run PRMHSyn on ALMANAC:

```bash
python Model/PRMHSyn.py --dataset ALMANAC --threshold 10 --split_method cell_line --learning_rate 5e-4 --weight_decay 5e-5 --epochs 2000
```

Replace `cell_line` with `cell_line_leaveout` or `drug_combo_leaveout` to use another evaluation setting.
