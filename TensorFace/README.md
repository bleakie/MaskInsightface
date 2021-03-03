## insightface TensorRT转换 ##

### 0.安装

```
tensorrt>=7.2
```

### 1.模型转换

参见InsightFace-REST：[**InsightFace-REST**](https://github.com/SthPhoenix/InsightFace-REST)


### List of supported models:

#### Detection:

| Model                 | Auto download | Inference code | Source                                                                                          |
|:----------------------|:--------------|:---------------|:------------------------------------------------------------------------------------------------|
| retinaface_r50_v1     | Yes           | Yes            | [official package](https://github.com/deepinsight/insightface/tree/master/python-package)       |
| retinaface_mnet025_v1 | Yes           | Yes            | [official package](https://github.com/deepinsight/insightface/tree/master/python-package)       |
| retinaface_mnet025_v2 | Yes           | Yes            | [official package](https://github.com/deepinsight/insightface/tree/master/python-package)       |
| mnet_cov2             | No            | Yes            | [mnet_cov2](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFaceAntiCov) |
| centerface            | Yes           | Yes            | [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace)                             |

#### Recognition:

| Model                  | Auto download | Inference code | Source                                                                                                    |
|:-----------------------|:--------------|:---------------|:----------------------------------------------------------------------------------------------------------|
| arcface_r100_v1        | Yes           | Yes            | [official package](https://github.com/deepinsight/insightface/tree/master/python-package)                 |
| r100-arcface-msfdrop75 | No            | Yes            | [SubCenter-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/SubCenter-ArcFace) |
| r50-arcface-msfdrop75  | No            | Yes            | [SubCenter-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/SubCenter-ArcFace) |
| glint360k_r100FC_1.0   | No            | Yes            | [Partial-FC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)               |
| glint360k_r100FC_0.1   | No            | Yes            | [Partial-FC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)               |

#### Other:

| Model        | Auto download | Inference code | Source                                                                                          |
|:-------------|:--------------|:---------------|:------------------------------------------------------------------------------------------------|
| genderage_v1 | Yes           | Yes            | [official package](https://github.com/deepinsight/insightface/tree/master/python-package)       |
| 2d106det     | No            | No             | [coordinateReg](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg) |


### 2.训练
#### 运行测试

```
python demo.py
```


#### 2.1.模型下载


 |      Backbone       |     差距    |  Speed  | Download |
 
 |---------------------|--------- --|---------|----------|
 
 |glint360k_r100FC_1.0 |   0.0001   |    8×   |[**code:ty5u**]() |

