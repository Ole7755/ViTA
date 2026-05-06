## 任务
- 三分类(睁眼、闭眼、无关)
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

### 无关数据集使用imagenet

## Model

### convnext_tiny
- input size: [B, 3, 128, 128]
- param: 27.82M

#### 微调
- 数据预处理: Resize((128, 128)) , RandomHorizontalFlip(p=0.5), ToTensor(), Normalize(ImageNet mean/std)
- 模型架构只改了模型分类头
- epochs: 5
- lr: 0.0005

#### 新构建的数据集
|Split|	closed|	open|	irrelevant|
|train|	3,986|	3,986|	3,986|
|val|	997|	997|	997|
|test|	2,135|	2,135|	2,135|

- irrelevant使用的是FoodSeg103数据集的图像

##### 在构建的数据集上的性能
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|eys_food| 0.9902  |  144.93 | 20.98|
##### 在全部数据集上的性能
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9860   |   | |
|OCE| 0.9764  |   |  |
##### 转为ONNX部署
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|eys_food| 0.9902  |   | 39.06|

|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9860   |   | |
|OCE| 0.9764  |   |  |

- 转FP16精度 FPS:9.78

## 指标
- FPS 每秒评测多少图像 (至少10帧 )
- ACC
- 测试GPU/CPU的性能

## 远程Linux信息
- GPU: RTX3090 24GB
- CPU: 16 核 AMD EPYC 7542

## 本地
- CPU: M4 Pro 