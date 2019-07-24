insightface制作自己的数据及其训练
# 1.生成，*.lst，property 
运行：sh im2rec_my_lst.sh生成*.lst （加上--train-ratio 0.9可以按比例划分train和val）
property是属性文件，里面内容是类别数和图像大小，例如 
1000,112,112 其中1000代表人脸的类别数目，图片格式为112x112（一直都不知道怎么自动生成这个，我是自己写的）

# 2.生成.rec，.idx
运行：sh im2rec_my.sh


# 3.生成测试文件.bin
## 3.1.用gen_valdatasets.py
直接修改里面的几个参数就ok（推荐）

## 3.2.使用sh文件
首先需要制作一个test.txt文件，格式如下：
路径/img_1.jpg,路径/img_2.jpg, 1
路径/img_3.jpg,路径/img_4.jpg, 0
[图片1,图片2,空格，标签]：如果相同，标签=1,不相同=0
