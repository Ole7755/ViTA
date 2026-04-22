## 任务
- 二分类(睁眼、闭眼)
## Dataset
### MRL Eye Dataset
- 数据集大小: 345M
- Train: Awake (25,770), Sleepy (25,167)
- Validation: Awake (8,591), Sleepy (8,389)
- Test: Awake (8,591), Sleepy (8,390)
### open-closed-eyes-dataset
- 数据集大小: 1.5G
- Train: (139804 images)
- Validation: (27960 images)
- Test: (6990 images)
## Model
### Swin-T
- input size: 224 * 224 * 3
- param: 29M
- ACC:
- FPS:
### Swin-S
- input size: 224 * 224 * 3
- param: 50M
- ACC:
- FPS:
### Swin-B
- input size: 224 * 224 * 3
- param: 88M
- ACC:
- FPS:


- 保留CNN模型做对比
## 指标
-  每秒评测多少图像 (至少10帧 )
- 关注几个vit模型
- 测试GPU/CPU的性能

## 远程Linux信息
- GPU: RTX3090 24GB
