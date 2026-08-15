# ViTA Eye State Classifier

ViTA 是一个用于眼部状态识别的三分类训练项目。模型输入眼部或非眼部图像，输出以下类别：

| Label | Class | 含义 |
| :-: | :- | :- |
| 0 | `closed` | 闭眼 |
| 1 | `open` | 睁眼 |
| 2 | `irrelevant` | 无关图像 |

当前默认模型使用 `convnext_tiny` 作为 backbone，输入图像统一 resize 到 `128 x 128`。训练方式是加载 ImageNet pretrained 权重，替换三分类分类头后，对 backbone 和 classifier head 一起 fine-tune。

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

当前默认训练使用已整合的三分类 ImageFolder 数据集，并额外加入 badcase 数据集；
`dataset4` 只作为真实场景测试集，不参与训练。
服务器路径分别为 `/home/featurize/data/blink_dataset`、
`/home/featurize/data/badcase/case1_dataset` 和
`/home/featurize/data/badcase/case2_dataset`。badcase 只包含 `closed` 和 `open`，
会沿用标签 0 和 1；训练、验证和测试三个 split 都会被合并读取。
本地 badcase 原始目录对应为 `/Volumes/Elements/datasets/blink_data/badcase`，
上传到服务器后应保持 `case1_dataset`、`case2_dataset` 两级目录结构。

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
├── manifest.csv
├── metadata.json
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
| Input size | `[B, 3, 128, 128]` |
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
Resize((128, 128))
-> RandomRotation(±110°, p=0.5)
-> RandomHorizontalFlip(p=0.5)
-> optional Gaussian blur
-> optional JPEG compression
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
Resize((128, 128)) -> ToTensor() -> ImageNet mean/std normalize
```

## 配置文件

默认配置位于：

```text
configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml
```

关键字段：

```yaml
experiment_name: eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01

data:
  image_size: 128
  batch_size: 128
  eval_batch_size: 256
  train_augment:
    rotation_degrees: 110
    rotation_p: 0.50
    gaussian_blur_p: 0.35
    gaussian_blur_kernel_size: 7
    gaussian_blur_sigma: [0.1, 2.0]
    jpeg_compression_p: 0.35
    jpeg_quality: [25, 85]
    random_erasing_p: 0.20
    random_erasing_scale: [0.02, 0.12]
  datasets:
    - name: blink_dataset
      root: /home/featurize/data/blink_dataset
      class_map:
        closed: 0
        open: 1
        irrelevant: 2
    - name: badcase_case1
      root: /home/featurize/data/badcase/case1_dataset
      class_map:
        closed: 0
        open: 1
    - name: badcase_case2
      root: /home/featurize/data/badcase/case2_dataset
      class_map:
        closed: 0
        open: 1

model:
  name: convnext_tiny
  pretrained: true
  kwargs:
    drop_path_rate: 0.1

train:
  epochs: 40
  warmup_epochs: 4
  lr: 0.00005
  weight_decay: 0.05
  class_weight: false
  report_dataset_metrics: true
  save_best_dataset_metrics: true
  selection_metric: mean_closed_open
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0005
```

默认配置已指向新 Linux 服务器上的整合数据集。迁移到其他机器时，通常只需要修改
`data.datasets[*].root`。

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
mkdir -p outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01

nohup python3 -m vita.train \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  > outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/train.log 2>&1 &
```

查看日志：

```bash
tail -f outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/train.log
```

训练输出会保存在：

```text
outputs/<experiment_name>/
├── best.pt       # selection metric 最优 checkpoint
├── last.pt       # 最后一轮 checkpoint
├── config.json   # 本次训练使用的配置
├── metrics.csv   # 每轮总 train/val 及各数据集 loss、ACC、分类别 ACC 和耗时
└── best_dataset_metrics.json  # best.pt 在各数据集 train/val/test 上的最终指标
```

当 `train.report_dataset_metrics` 为 `true` 时，训练日志会额外输出
`blink_dataset`、`badcase_case1` 和 `badcase_case2` 各自的 train/val 指标。
这些指标使用不带随机增强的确定性预处理；它们只读取训练配置中的数据集，
不会读取独立的 `dataset4` 测试目录。

当 `train.save_best_dataset_metrics` 为 `true` 时，训练结束会重新加载
`best.pt`，在 `blink_dataset`、`badcase_case1` 和 `badcase_case2` 的
`train`、`val`、`test` split 上做确定性评估。结果会同时写入
`best.pt` 的 `metrics.dataset_metrics` 和 `best_dataset_metrics.json`；
`dataset4` 不参与这一步。

如果 early stopping 被触发，训练会提前结束，但 `best.pt` 仍然保存验证指标最优的 checkpoint。

## 导出 ONNX

训练完成后，可以将验证集表现最好的 `best.pt` 导出为 ONNX。导出脚本会同时生成 metadata JSON，记录类别名、输入输出节点名和推理前预处理参数。

```bash
mkdir -p exports

python3 -m vita.export_onnx \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/best.pt \
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

## 评测

### PyTorch checkpoint 评测

```bash
python3 -m vita.eval \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/best.pt \
  --split test \
  --output outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/test_metrics.json
```

### ONNXRuntime 评测

新服务器上的默认独立测试集为 `/home/featurize/data/test`，其中包含
`cam0`、`cam1`、`dataset4`、`img` 四个子集。可使用专用脚本自动逐个评测并汇总：

```bash
python3 scripts/eval_dataset4.py
```

也可以使用通用评测入口并显式指定该目录：

```bash
python3 -m vita.eval \
  --onnx exports/eye_state_classification.onnx \
  --metadata exports/eye_state_classification.json \
  --data-root /home/featurize/data/test \
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
    "resize": [128, 128],
    "interpolation": "bilinear",
    "antialias": true,
    "color_order": "RGB",
    "input_scale": 255.0,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

部署时需要保证应用侧预处理与 metadata 中的 `resize`、`interpolation`、`color_order`、`input_scale`、`mean`、`std` 完全一致。类别顺序固定为：

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
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/best.pt \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --output outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/benchmark_cuda.json
```

CPU 吞吐量：

```bash
python3 -m vita.benchmark \
  --config configs/eye3_mixed_unknown_convnext_tiny_64_aug.yaml \
  --checkpoint outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/best.pt \
  --device cpu \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads 16 \
  --output outputs/eye3_mixed_plus_eye_dataset_plus_badcase_convnext_tiny_128_drop_path_01/benchmark_cpu.json
```
