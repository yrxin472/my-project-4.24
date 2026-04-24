# EuroSAT MLP Homework 1

这是一个**完全不依赖 PyTorch / TensorFlow / JAX** 的 NumPy 版三层 MLP 图像分类作业实现，严格对应题目要求：

- 自主实现自动微分与反向传播
- 数据加载与预处理
- 模型定义
- 训练循环
- 测试评估
- 超参数搜索
- 第一层权重可视化
- 错例分析
- 自动生成实验报告

## 1. 环境依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows 请改为 .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 数据集组织方式

请将 EuroSAT 数据集整理为如下结构（即题目中提到的 `EuroSAT_RGB` 文件夹）：

```text
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

每个类别文件夹下放对应的 RGB 图像。

## 3. 一键训练

```bash
python train.py \
  --data_root /path/to/EuroSAT_RGB \
  --output_dir runs/baseline \
  --epochs 30 \
  --batch_size 128 \
  --hidden_dim 256 \
  --activation relu \
  --lr 0.05 \
  --weight_decay 1e-4 \
  --lr_step 10 \
  --lr_gamma 0.5
```

训练完成后会自动保存：

- `history.json`
- `best_model.npz`
- `config.json`
- `stats.json`
- `splits.json`
- `training_curves.png`

## 4. 超参数搜索

### 网格搜索
```bash
python search.py \
  --data_root /path/to/EuroSAT_RGB \
  --search_type grid \
  --output_dir runs/search_grid \
  --hidden_dims 128,256 \
  --activations relu,tanh \
  --lrs 0.1,0.05,0.01 \
  --weight_decays 0.0,1e-4 \
  --epochs 20
```

### 随机搜索
```bash
python search.py \
  --data_root /path/to/EuroSAT_RGB \
  --search_type random \
  --num_trials 12 \
  --output_dir runs/search_random \
  --hidden_dims 128,256,384,512 \
  --activations relu,tanh,sigmoid \
  --lrs 0.1,0.05,0.02,0.01 \
  --weight_decays 0.0,1e-5,1e-4,5e-4 \
  --epochs 20
```

搜索结果会保存为：

- `search_results.csv`
- `search_results.json`
- `best_config.json`

## 5. 测试与评估

```bash
python test.py --experiment_dir runs/baseline
```

会输出：

- 测试集 Accuracy
- `confusion_matrix.txt`
- `confusion_matrix.png`
- `test_metrics.json`

## 6. 权重可视化与错例分析

```bash
python visualize.py --experiment_dir runs/baseline
```

会输出：

- `first_layer_weights.png`
- `misclassified_samples.png`

## 7. 自动生成实验报告

```bash
python tools/build_report.py --experiment_dir runs/baseline
```

将自动生成：

- `experiment_report.docx`

如本地装有 LibreOffice，也可自行转 PDF：

```bash
python /home/oai/skills/docx/render_docx.py runs/baseline/experiment_report.docx --output_dir runs/baseline/report_render --emit_pdf
```

## 8. 代码结构

```text
eurosat_mlp_hw1/
├── train.py
├── search.py
├── test.py
├── visualize.py
├── requirements.txt
├── README.md
├── src/
│   ├── autograd.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── optim.py
│   ├── trainer.py
│   ├── utils.py
│   └── visualization.py
└── tools/
    └── build_report.py
```

## 9. 说明

1. 本项目中的三层网络指 **三层全连接层：FC1 -> FC2 -> FC3**，其中前两层后接激活函数。
2. 输入图像默认 resize 到 `64×64`，并展平成向量送入 MLP。
3. 为保证结果可复现，代码默认使用固定随机种子，并将数据划分信息写入 `splits.json`。
4. 报告模板中包含 GitHub Repo 链接与权重下载地址的占位符，需要你在提交前替换成自己的真实链接。
