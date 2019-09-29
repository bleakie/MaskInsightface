## insightface制作自己的数据及其训练 ##

说明：算法集成insightface：https://github.com/deepinsight/insightface

改进地方：ssh(人脸检测)+prnet(68 landmark 人脸对齐， 3d人脸mask)+insightface

```
(1)修改人脸检测器（优化后的SSH，误检率更低，对大角度和blur的face进行过滤）
(2)使用68个点的landmark，prnet的对齐效果更准
(3)利用prnet拟合有效区域的人脸位置，抠出背景，以平均人脸像素填充背景，减少噪声影响，会在图片质量较好的情况下提高识别
```

![MASK0](https://github.com/bleakie/MaskInsightface/blob/master/images/Akbar_Al_Baker_0001.jpg)

![MASK1](https://github.com/bleakie/MaskInsightface/blob/master/images/Abner_Martinez_0001.jpg)

### 0.安装

```
(1)mxnet
(2)tensorflow
```

### 1.生成对齐后的数据集

#### 1.1.数据下载

http://trillionpairs.deepglint.com/data

cd make_rec

#### 1.2.生成,'.lst, .rec, .idx, property'

(1)为了合并数据，可采用generate_lst.sh

(2)property是属性文件，里面内容是类别数和图像大小，例如

1000,112,112 其中1000代表人脸的类别数目，图片格式为112x112（一直都不知道怎么自动生成这个，我是自己写的）

(3)generate_lst.sh

#### 1.3.生成测试文件.bin

```
python gen_valdatasets.py
```

#### 1.4.生成数据

```
python3 gen_datasets.py  #完成后会output下生成train.lst
```
### 2.验证model精度

#### 2.1.在bash

```
python3 -u ./src/eval/verification.py --gpu 0 --model "./models/glint-mobilenet/model,1" --target 'lfw'
```

#### 2.2.快捷

```
sh verification.sh
```

### 3.训练

#### 3.1.在bash里面训练

```
CUDA_VISIBLE_DEVICES='2,3,4,5' python3 -u train.py --network r100 --loss arcface --per-batch-size 64 2>&1 > log.log &
```

#### 3.2.如果想要合并不同数据集

```
CUDA_VISIBLE_DEVICES=0 python3 src/data/dataset_merge.py --include 001_data,002_data --output ms1m+vgg --model ../../models/model,1
```

### 4.result
'参数设置'
network backbone: r100 ( output=E, emb_size=512, prelu )

loss function: arcface(m=0.5)

batch-size:256, 4gpu, config.fc7_wd_mult = 10
 
lr = 0.004, lr_steps [105000, 125000, 150000], default.wd = 0.0005, end with 180001,

then retrain with lr = 0.0004, lr_steps[200000, 300000, 400000], default.wd = 0.00001


|  Data    |      LFW   |    CFP_FP    |  AgeDB30  |
| -------- | -----------|--------------|---------- |
|  ACCU(%) |    99.82+  |    98.50+    |  98.25+   |

## Todo
释放训练好的模型（SSH,PRNET,人脸识别）
