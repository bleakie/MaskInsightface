# SSR-Net
**[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation**
+ A real-time age estimation model with 0.32MB.
+ Gender regression is also added!
+ Megaage-Asian is provided in https://github.com/b02901145/SSR-Net_megaage-asian
+ Coreml model (0.17MB) is provided in https://github.com/shamangary/Keras-to-coreml-multiple-inputs-example

**Code Author: Tsun-Yi Yang**

## Paper

### PDF
https://github.com/shamangary/SSR-Net/blob/master/ijcai18_ssrnet_pdfa_2b.pdf

### Paper authors
**[Tsun-Yi Yang](http://shamangary.logdown.com/), [Yi-Husan Huang](https://github.com/b02901145), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), [Pi-Cheng Hsiu](https://www.citi.sinica.edu.tw/pages/pchsiu/index_en.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

## Abstract
This paper presents a novel CNN model called Soft Stagewise Regression Network (SSR-Net) for age estimation from a single image with a compact model size. Inspired by DEX, we address age estimation by performing multi-class classification and then turning classification results into regression by calculating the expected values. SSR-Net takes a coarse-to-fine strategy and performs multi-class classification with multiple stages. Each stage is only responsible for refining the decision of the previous stage. Thus, each stage performs a task with few classes and requires few neurons, greatly reducing the model size. For addressing the quantization issue introduced by grouping ages into classes, SSR-Net assigns a dynamic range to each age class by allowing it to be shifted and scaled according to the input face image. Both the multi-stage strategy and the dynamic range are incorporated into the formulation of soft stagewise regression. A novel network architecture is proposed for carrying out soft stagewise regression. The resultant SSR-Net model is very compact and takes only **0.32 MB**. Despite of its compact size, SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are more than 1500x larger.

## Test
```
python3 TYY_demo_centerface_bbox_age_gender.py
```


## Result 
 | gender(%)     |  age MAE       |  age  RMSE         |  Speed(ms) |
|--------------|-----------|--------------|----------|
|98.53      | 3.29      |    4.25      |  3 |

```

## Third Party Implementation
+ MXNET:
https://github.com/wayen820/gender_age_estimation_mxnet

+ Pytorch:
https://github.com/oukohou/SSR_Net_Pytorch

+ Pytorch:
https://github.com/CrazySummerday/SSR-Net
