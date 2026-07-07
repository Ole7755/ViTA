# ViTA Eye State Classifier

ViTA 是一个用于眼部状态识别的三分类训练项目。模型输入眼部或非眼部图像，输出以下类别：

| Label | Class | 含义 |
| :-: | :- | :- |
| 0 | `closed` | 闭眼 |
| 1 | `open` | 睁眼 |
| 2 | `irrelevant` | 无关图像 |

当前默认模型使用 `convnext_tiny` 作为 backbone，输入图像统一 resize 到 `64 x 64`。训练方式是加载 ImageNet pretrained 权重，替换三分类分类头后，对 backbone 和 classifier head 一起 fine-tune。

## 项目结构

```text
.
├── configs/
│   └── eye3_mixed_unknown_convnext_tiny_64_aug.yaml
├── vita/
│   ├── benchmark.py      # 随机输入吞吐量测试，输出 FPS
│   ├── data.py           # ImageFolder-style 数据集读取与预处理
│   ├── eval.py           # PyTorch / ONNXRuntime ACC、confusion matrix、FPS 评测
│   ├── export_onnx.py    # 将 PyTorch checkpoint 导出为 ONNX
│   ├── models.py         # timm 模型构建
│   ├── train.py          # 训练、early stopping 与 checkpoint 保存
│   └── utils.py
├── requirements.txt
└── README.md
```

## 数据集

项目任务来自眼部开闭状态识别，并额外加入 `irrelevant` 作为第三类，用于降低模型在非眼部输入上的误判。

### 原始数据来源

| Dataset | Size | Train | Validation | Test |
| :- | :-: | :- | :- | :- |
| MRL Eye Dataset | 345M | Awake 25,770 / Sleepy 25,167 | Awake 8,591 / Sleepy 8,389 | Awake 8,591 / Sleepy 8,390 |
| open-closed-eyes-dataset | 1.5G | 139,804 images | 27,960 images | 6,990 images |
| COCO images | - | 用作 `irrelevant` 类的非眼部图像来源 | - | - |

当前训练使用整理后的三分类 ImageFolder 数据集。基础数据集为 `eye3_mixed_unknown`，新增泛化数据集为 `eye_dataset_split`。

### 数据下载

| Dataset | 说明 | 网盘链接 | 提取码 |
| :- | :- | :- | :-: |
| MRL Eye Dataset | 原始眼部开闭数据集 | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |
| open-closed-eyes-dataset | 原始眼部开闭数据集 | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |
| eye3_mixed_unknown | 已整理好的三分类数据集，包含 `closed`、`open`、`irrelevant` | [百度网盘](https://pan.baidu.com/s/1zk2x7KSyjpp6rdsc6tgkuA?pwd=bppr) | `bppr` |
| eye_dataset_split | 新增三分类数据集，包含 `train`、`val`、`test` | [百度网盘](https://pan.baidu.com/s/1zMV_ZkbRtlSlaW7MWiyx9w) | `pspi` |

### 数据统计

`eye3_mixed_unknown`：

| Split | `closed` | `open` | `irrelevant` |
| :-: | :-: | :-: | :-: |
| train | 7486 | 7486 | 7486 |
| val | 1747 | 1747 | 1747 |
| test | 2885 | 2885 | 2885 |

`eye_dataset_split`：

| Split | `closed` | `open` | `irrelevant` |
| :-: | :-: | :-: | :-: |
| train | 18743 | 17891 | 14681 |
| val | 3244 | 3186 | 2183 |
| test | 5627 | 5717 | 4396 |

### 目录格式

训练脚本读取 ImageFolder 风格的数据目录。每个训练数据集根目录下需要包含 `train`、`val`、`test` 三个 split，每个 split 下包含类别目录：

```text
<dataset_root>/
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

如果使用 MRL 的原始类名，可以在配置中将 `sleepy` 映射为 `closed`，将 `awake` 映射为 `open`。评测脚本还支持将 `close` 目录名映射为 `closed`。

## 环境准备

建议使用 Python 3.10+。GPU 训练时请先根据 CUDA 版本安装匹配的 PyTorch 和 torchvision。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖：

```text
torch
torchvision
timm
PyYAML
tqdm
Pillow
numpy
onnx
onnxscript
onnxruntime
```

## 模型与训练策略

| Item | Setting |
| :- | :- |
| Backbone | `convnext_tiny` |
| Input size | `[B, 3, 64, 64]` |
| Initialization | ImageNet pretrained |
| Fine-tuning | backbone 和 classifier head 一起训练 |
| Optimizer | AdamW |
| LR | `5e-5` |
| Weight decay | `0.05` |
| Epochs | `40` |
| Warmup | `4` epochs |
| AMP | enabled on CUDA |
| Class weight | disabled |
| Selection metric | `mean_closed_open` |
| Early stopping | patience `8`, min delta `0.0005` |

`mean_closed_open` 的定义为：

```text
(closed_acc + open_acc) / 2
```

这个指标用于选择 `best.pt`，目的是让模型同时兼顾 `closed` 和 `open`，避免选择标准偏向单一类别。

## 数据增强

训练集使用随机增强，验证集和测试集只做确定性预处理。

通用训练预处理流程：

```text
Resize((64, 64))
-> RandomRotation(±30°)
-> RandomHorizontalFlip(p=0.5)
-> class-specific color augmentation
-> ToTensor()
-> optional RandomErasing
-> ImageNet mean/std normalize
```

当前增强策略：

| Class | 策略 |
| :- | :- |
| `closed` | 较强增强：蓝紫偏色、随机灰度、低对比度/亮度/饱和度扰动、随机 patch 遮挡 |
| `open` | 弱化增强：降低蓝紫偏色、灰度、低对比度和随机 patch 遮挡概率 |
| `irrelevant` | 只使用全局旋转和水平翻转，不使用眼部类别增强 |

Val/Test 预处理流程：

```text
Resize((64, 64)) -> ToTensor() -> ImageNet mean/std normalize
```

## 配置文件

默认配置位于：

```text
configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml
```

关键字段：

```yaml
experiment_name: eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess

data:
  image_size: 64
  batch_size: 512
  eval_batch_size: 1024
  train_augment:
    rotation_degrees: 30
  datasets:
    - root: /path/to/eye3_mixed_unknown
      class_map:
        closed: 0
        open: 1
        irrelevant: 2
    - root: /path/to/eye_dataset_split
      class_map:
        closed: 0
        open: 1
        irrelevant: 2

train:
  epochs: 40
  warmup_epochs: 4
  lr: 0.00005
  weight_decay: 0.05
  class_weight: false
  selection_metric: mean_closed_open
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0005
```

复现训练时通常只需要修改 `data.datasets[*].root`，让它们指向本机或远程机器上的数据集目录。

## 训练

训练需要以 module 方式启动，因为脚本使用了包内相对导入：

```bash
python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml
```

如需强制使用 CPU：

```bash
python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --device cpu
```

后台训练并保存日志：

```bash
mkdir -p outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess

nohup python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  > outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/train.log 2>&1 &
```

查看日志：

```bash
tail -f outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/train.log
```

训练输出会保存在：

```text
outputs/<experiment_name>/
├── best.pt       # selection metric 最优 checkpoint
├── last.pt       # 最后一轮 checkpoint
├── config.json   # 本次训练使用的配置
└── metrics.csv   # 每轮 train/val loss、ACC 和耗时
```

如果 early stopping 被触发，训练会提前结束，但 `best.pt` 仍然保存验证指标最优的 checkpoint。

## 导出 ONNX

训练完成后，可以将验证集表现最好的 `best.pt` 导出为 ONNX。导出脚本会同时生成 metadata JSON，记录类别名、输入输出节点名和推理前预处理参数。

```bash
mkdir -p exports

python3 -m vita.export_onnx \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/best.pt \
  --output exports/eye_state_classification.onnx \
  --metadata-output exports/eye_state_classification.json \
  --device cpu \
  --opset 17
```

导出的 ONNX 模型输入输出约定如下：

| Item | Value |
| :- | :- |
| Input name | `images` |
| Input shape | `[B, 3, 64, 64]` |
| Output name | `logits` |
| Output shape | `[B, 3]` |
| Dynamic batch | 支持 |
| Precision | FP32 |

## 评测

### PyTorch checkpoint 评测

```bash
python3 -m vita.eval \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/best.pt \
  --split test \
  --output outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/test_metrics.json
```

### ONNXRuntime 评测

```bash
python3 -m vita.eval \
  --onnx exports/eye_state_classification.onnx \
  --metadata exports/eye_state_classification.json \
  --data-root /path/to/eval_dataset \
  --split test \
  --batch-size 512 \
  --warmup-batches 5 \
  --output outputs/onnx_eval_metrics.json
```

`--data-root` 支持两种 ImageFolder 风格目录。

单个数据集：

```text
/path/to/eval_dataset/
├── closed/
├── open/
└── irrelevant/
```

或多个子数据集：

```text
/path/to/eval_dataset/
├── subset_a/
│   ├── closed/
│   ├── open/
│   └── irrelevant/
└── subset_b/
    ├── closed/
    ├── open/
    └── irrelevant/
```

评测 JSON 会包含以下核心字段：

- `accuracy`
- `per_class_accuracy`
- `confusion_matrix`
- `class_names`
- `num_samples`
- `total_seconds`
- `inference_seconds`
- `samples_per_second`
- `inference_samples_per_second`

如果输入目录包含多个子数据集，还会包含 `num_subsets` 和 `subsets`。

## 部署预处理约定

metadata JSON 示例：

```json
{
  "class_names": ["closed", "open", "irrelevant"],
  "input_name": "images",
  "output_name": "logits",
  "preprocess": {
    "resize": [64, 64],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

部署时需要保证应用侧预处理与 metadata 中的 `resize`、`mean`、`std` 完全一致。类别顺序固定为：

```text
0 -> closed
1 -> open
2 -> irrelevant
```

## 权重下载

| 权重 | 说明 | 网盘链接 | 提取码 |
| :- | :- | :- | :-: |
| PyTorch checkpoint | `.pt` 格式模型权重，可用于继续微调或 PyTorch 评测 | [百度网盘](https://pan.baidu.com/s/1hVKfuQNuDsXgJvhQnpANqQ?pwd=8b86) | `8b86` |
| ONNX model | `.onnx` 格式模型权重，可用于 ONNXRuntime 部署 | [百度网盘](https://pan.baidu.com/s/1hLTIT_l0Is_KvmFaLL3hVA?pwd=btza) | `btza` |

## 测试 FPS

FPS 指每秒可评测多少张图像。为了让结果更稳定，建议保证 `batch_size * steps >= 10`，并在固定硬件、固定线程数下比较。

GPU 吞吐量：

```bash
python3 -m vita.benchmark \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/best.pt \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --output outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/benchmark_cuda.json
```

CPU 吞吐量：

```bash
python3 -m vita.benchmark \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/best.pt \
  --device cpu \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads 16 \
  --output outputs/eye3_mixed_plus_eye_dataset_convnext_tiny_64_aug_preprocess/benchmark_cpu.json
```
