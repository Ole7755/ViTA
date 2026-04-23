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
### swin_tiny_patch4_window7_224
- input size: [B, 3, 224, 224]
- param: 27.52M

#### 微调
- 数据预处理: Resize((224, 224)) , RandomHorizontalFlip(p=0.5), ToTensor(), Normalize(ImageNet mean/std)
- 模型架构只改了模型分类头
- epochs: 5
- lr: 0.0005

##### 在MRL微调
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9902   |   94.04 | 5.92 |
|OCE| 0.9651   |  93.73 | 5.97  |

#### 在本地上笔记本上的性能
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9902  |   33.31| 33.31|
|OCE| 0.9651   |  32.84| 32.84  |


### convnext_tiny
- input size: [B, 3, 224, 224]
- param: 27.82M

#### 微调
- 数据预处理: Resize((224, 224)) , RandomHorizontalFlip(p=0.5), ToTensor(), Normalize(ImageNet mean/std)
- 模型架构只改了模型分类头
- epochs: 5
- lr: 0.0005

##### 在MRL微调
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9923   |  136.56 | 7.24|
|OCE| 0.9774   |  142.69 | 10.60  |

#### 在本地上笔记本上的性能
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9902  |   3.16| 3.16|
|OCE| 0.9651   |  3.18| 3.18  |

### ResNet-50
- input size: [B, 3, 224, 224]
- param: 23.51M

#### 微调
- 数据预处理: Resize((224, 224)) , RandomHorizontalFlip(p=0.5), ToTensor(), Normalize(ImageNet mean/std)
- 模型架构只改了模型分类头
- epochs: 5
- lr: 0.0005

##### 在MRL微调
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9899    |  148.00 | 7.01|
|OCE| 0.9394    |  144.53 | 6.10 |

#### 在本地上笔记本上的性能
|Dataset|ACC|FPS|CPU FPS|
|:-:|:-:|:-:|:-:|
|MRL| 0.9902  |   46.26| 46.26|
|OCE| 0.9651   |  44.87| 44.87  |

## 指标
- FPS 每秒评测多少图像 (至少10帧 )
- ACC
- 测试GPU/CPU的性能

## 远程Linux信息
- GPU: RTX3090 24GB
- CPU: 16 核 AMD EPYC 7542

## 本地
- CPU: M4 Pro (16+20)