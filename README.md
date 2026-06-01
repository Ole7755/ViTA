# ViTA Eye State Classifier

ViTA 是一个用于眼部状态识别的三分类训练项目，目标类别为：

| Label | Class | 含义 |
| :-: | :- | :- |
| 0 | `closed` | 闭眼 |
| 1 | `open` | 睁眼 |
| 2 | `irrelevant` | 无关图像 |

当前仓库以 `convnext_tiny` 为主干模型，将输入图像统一 resize 到 `128 x 128`。模型不是从零重新训练，而是加载 ImageNet pretrained 权重，替换为三分类分类头后，将主干网络和分类头一起 fine-tune。项目包含数据加载、训练、评测和吞吐量测试脚本，便于后续复现训练流程或继续微调模型。

## 项目结构

```text
.
├── configs/
│   └── eye3_mixed_unknown_convnext_tiny_128_aug.yaml
├── vita/
│   ├── benchmark.py      # 随机输入吞吐量测试，输出 FPS
│   ├── data.py           # ImageFolder-style 数据集读取与预处理
│   ├── eval.py           # ACC、per-class ACC、confusion matrix 评测
│   ├── export_onnx.py    # 将 PyTorch checkpoint 导出为 ONNX
│   ├── models.py         # timm 模型构建
│   ├── train.py          # 训练与 checkpoint 保存
│   └── utils.py
├── requirements.txt
└── README.md
```

## 数据集

项目任务来自眼部开闭状态识别，并额外加入 `irrelevant` 作为第三类，减少模型在非眼部输入上的误判。

### 原始数据来源

| Dataset | Size | Train | Validation | Test |
| :- | :-: | :- | :- | :- |
| MRL Eye Dataset | 345M | Awake 25,770 / Sleepy 25,167 | Awake 8,591 / Sleepy 8,389 | Awake 8,591 / Sleepy 8,390 |
| open-closed-eyes-dataset | 1.5G | 139,804 images | 27,960 images | 6,990 images |
| COCO images | - | 用作 `irrelevant` 类的非眼部图像来源 | - | - |

当前记录的三分类实验中，`irrelevant` 类使用 COCO 数据集图像构建。

### 数据下载

MRL Eye Dataset、open-closed-eyes-dataset 以及项目自制的三分类数据集会统一放在网盘中，后续需要复现训练或继续微调时可以直接下载。

| Dataset | 说明 | 网盘链接 | 提取码 |
| :- | :- | :- | :-: |
| MRL Eye Dataset | 原始眼部开闭数据集 | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |
| open-closed-eyes-dataset | 原始眼部开闭数据集 | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |
| eye3_mixed_unknown | 已整理好的三分类数据集，包含 `closed`、`open`、`irrelevant` | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |

### 构建后的三分类数据

| Split | `closed` | `open` | `irrelevant` |
| :-: | :-: | :-: | :-: |
| train | 7486 | 7486 | 7486 |
| val | 1747 | 1747 | 1747 |
| test | 2885 | 2885 | 2885 |

### 目录格式

训练脚本读取 ImageFolder 风格的数据目录。每个数据集根目录下需要包含 `train`、`val`、`test` 三个 split，每个 split 下包含类别目录：

```text
/path/to/eye3_mixed_unknown/
├── train/
│   ├── closed/
│   ├── open/
│   └── irrelevant/
├── val/
│   ├── closed/
│   ├── open/
│   └── irrelevant/
└── test/
    ├── closed/
    ├── open/
    └── irrelevant/
```

如果使用 MRL 的原始类名，也可以在配置里将 `sleepy` 映射为 `closed`，将 `awake` 映射为 `open`。代码会在生成类别名时做这个别名转换。

## 环境准备

建议使用 Python 3.10+。GPU 训练时请先根据 CUDA 版本安装匹配的 PyTorch 和 torchvision。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

远程 Linux 参考环境：

| Hardware | Spec |
| :- | :- |
| GPU | RTX 3090 24GB |
| CPU | 16-core AMD EPYC 7542 |

## 模型与微调方式

| Item | Setting |
| :- | :- |
| Backbone | `convnext_tiny` |
| Input size | `[B, 3, 128, 128]` |
| Parameters | 约 27.82M |
| Initialization | ImageNet pretrained |
| Fine-tuning | 替换三分类分类头，主干网络和分类头一起训练 |

训练时不会冻结 `convnext_tiny` 主干参数；优化器会同时更新 backbone 和 classifier head。

## 配置文件

默认配置位于 `configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml`：

```yaml
experiment_name: eye3_mixed_unknown_convnext_tiny_128_aug_preprocess
output_dir: outputs
seed: 42

data:
  image_size: 128
  batch_size: 512
  eval_batch_size: 1024
  num_workers: 8
  pin_memory: true
  train_augment: {}
  train_augment_by_class:
    closed:
      rotation_degrees: 8
      blue_purple_p: 0.25
      blue_purple_strength: [0.08, 0.20]
      grayscale_p: 0.30
      low_contrast_p: 0.45
      brightness: [0.70, 1.15]
      contrast: [0.45, 1.0]
      saturation: [0.0, 1.0]
    open:
      rotation_degrees: 6
  datasets:
    - root: /home/featurize/data/eye3_mixed_unknown
      class_map:
        closed: 0
        open: 1
        irrelevant: 2

model:
  name: convnext_tiny
  pretrained: true

train:
  device: cuda
  epochs: 8
  lr: 0.00005
  weight_decay: 0.05
  warmup_epochs: 2
  amp: true
  class_weight: false
  class_weights:
    closed: 1.8
    open: 1.0
    irrelevant: 1.0
  selection_metric: min_closed_open_plus_0_3_closed
```

复现或微调时通常需要修改：

| Field | 说明 |
| :- | :- |
| `data.datasets[0].root` | 三分类数据集根目录 |
| `data.image_size` | 输入尺寸，当前为 `128` |
| `data.batch_size` | 训练 batch size |
| `data.eval_batch_size` | 验证/测试 batch size |
| `model.name` | timm 模型名，当前为 `convnext_tiny` |
| `train.epochs` | 训练轮数，当前为 `8` |
| `train.lr` | 当前配置学习率为 `5e-5` |
| `train.device` | `cuda` 或 `cpu` |

预处理与增强流程：

- Train: `Resize((128, 128))` -> 按类别增强 -> `RandomHorizontalFlip(p=0.5)` -> `ToTensor()` -> ImageNet mean/std normalize
- `closed`: 额外使用小角度旋转、蓝紫偏色、灰度化、低对比度和亮度/对比度/饱和度扰动
- `open`: 额外使用小角度旋转
- Val/Test: `Resize((128, 128))` -> `ToTensor()` -> ImageNet mean/std normalize

## 训练

训练需要以 module 方式启动，因为脚本使用了包内相对导入：

```bash
python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml
```

如需强制使用 CPU：

```bash
python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml \
  --device cpu
```

训练输出会保存在：

```text
outputs/<experiment_name>/
├── best.pt       # 验证集 ACC 最优 checkpoint
├── last.pt       # 最后一轮 checkpoint
├── config.json   # 本次训练使用的配置
└── metrics.csv   # 每轮 train/val loss、ACC 和耗时
```

## 导出 ONNX

训练完成后，可以将验证集表现最好的 `best.pt` 导出为 ONNX，用于 ONNXRuntime 或后续应用部署。导出脚本会同时生成一个 metadata JSON，记录类别名、输入输出节点名和推理前预处理参数。

```bash
python3 -m vita.export_onnx \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml \
  --checkpoint outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/best.pt \
  --output exports/eye_state_classification.onnx \
  --metadata-output exports/eye_state_classification.json \
  --device cpu \
  --opset 17
```

导出的 ONNX 模型输入输出约定如下：

| Item | Value |
| :- | :- |
| Input name | `images` |
| Input shape | `[B, 3, 128, 128]` |
| Output name | `logits` |
| Output shape | `[B, 3]` |
| Dynamic batch | 支持 |
| Precision | FP32 |

metadata JSON 中包含部署侧需要保持一致的预处理配置：

```json
{
  "class_names": ["closed", "open", "irrelevant"],
  "input_name": "images",
  "output_name": "logits",
  "preprocess": {
    "resize": [128, 128],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

注意：部署时需要保证应用侧预处理与 metadata 中的 `resize`、`mean`、`std` 完全一致，否则 ONNX 与 PyTorch 评测结果可能出现偏差。

## 权重下载

| 权重 | 说明 | 网盘链接 | 提取码 |
| :- | :- | :- | :-: |
| PyTorch checkpoint | `.pt` 格式模型权重，可用于继续微调或 PyTorch 评测 | [百度网盘](https://pan.baidu.com/s/1hVKfuQNuDsXgJvhQnpANqQ?pwd=8b86) | `8b86` |
| ONNX model | `.onnx` 格式模型权重，可用于 ONNXRuntime 部署 | [百度网盘](https://pan.baidu.com/s/1hLTIT_l0Is_KvmFaLL3hVA?pwd=btza) | `btza` |

## 评测 ACC

在 test split 上评测：

```bash
python3 -m vita.eval \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml \
  --checkpoint outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/best.pt \
  --split test \
  --output outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/test_metrics.json
```

评测脚本会打印总体 ACC，并可保存如下信息到 JSON：

- `accuracy`
- `per_class_accuracy`
- `confusion_matrix`
- `class_names`

## 测试 FPS

FPS 指每秒可评测多少张图像。为了让结果更稳定，建议保证 `batch_size * steps >= 10`，并在固定硬件、固定线程数下比较。

GPU 吞吐量：

```bash
python3 -m vita.benchmark \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml \
  --checkpoint outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/best.pt \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --output outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/benchmark_cuda.json
```

CPU 吞吐量：

```bash
python3 -m vita.benchmark \
  --config configs/eye3_mixed_unknown_convnext_tiny_128_aug.yaml \
  --checkpoint outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/best.pt \
  --device cpu \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads 16 \
  --output outputs/eye3_mixed_unknown_convnext_tiny_128_aug_preprocess/benchmark_cpu.json
```
