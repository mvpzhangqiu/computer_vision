# undermine_detect

**如果帮到您请给个 star！**

该项目是使用 `YOLOv5 v2.x` 的程序来训练在煤井下目标检测的应用。

---

# 1.YOLO v5 训练自己数据集教程

可使用的公开数据数据集：[Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) ，可用于人员以及安全帽佩戴检测。

> 本文结合 [YOLOv5 官方教程](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 来写

## 环境准备

首先确保自己的环境：

```text
    Python>=3.7
    Pytorch==1.5.x
    PyQt5==5.15.3
    PyQtChart==5.15.3
    PyQt5-tools
    GPUtil
```

或者使用我的环境（推荐）

```shell script
pip install -r requirements.txt
```

## 官方权重

我已上传到一份到百度云：[下载链接](https://pan.baidu.com/s/1mSIjDAzfiJd1fqSxIYzRDA) , 密码: `44qm`

## 训练自己的数据

#### 提示：

**关于增加数据集分类的方法，请看【5. 增加数据集的分类】**

---

### 1.1 创建自己的数据集配置文件

基于 `data/coco128.yaml` 文件，创建自己的数据集配置文件 `custom_data.yaml`

```yaml
# 训练集和验证集的 labels 和 image 文件的位置
train: [/XX/train/images/]
val: [/XX/val/images/]

# number of classes
nc: X

# class names
names: [XX, ...]
```

### 1.2 创建每个图片对应的标签文件

你可以使用 `data/gen_data/gen_head_helmet.py` 来将` VOC` 的数据集转换成 `YOLOv5` 训练需要用到的格式。

使用标注工具类似于 [Labelbox](https://labelbox.com/) 、[VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) 标注之后，需要生成每个图片对应的 `.txt` 文件，其规范如下：

- 每一行都是一个目标
- 类别序号是零索引开始的（从 0 开始）
- 每一行的坐标 `class x_center y_center width height` 格式
- 框坐标必须采用**归一化的 xywh**格式（从 0 到 1）。如果您的框以像素为单位，则将`x_center`和`width`除以图像宽度，将`y_center`和`height`除以图像高度。代码见`json2txt.py`。

生成的 `.txt` 文件放置的名字是图片的名字，放置在 label 文件夹中，例如：

```text
XX/images/train/00001.jpg  # image
XX/labels/train/00001.txt  # label
```

生成的 `.txt` 例子：

```text
1 0.1830000086920336 0.1396396430209279 0.13400000636465847 0.15915916301310062
1 0.5240000248886645 0.29129129834473133 0.0800000037997961 0.16816817224025726
1 0.6060000287834555 0.29579580295830965 0.08400000398978591 0.1771771814674139
1 0.6760000321082771 0.25375375989824533 0.10000000474974513 0.21321321837604046
0 0.39300001866649836 0.2552552614361048 0.17800000845454633 0.2822822891175747
0 0.7200000341981649 0.5570570705458522 0.25200001196935773 0.4294294398277998
0 0.7720000366680324 0.2567567629739642 0.1520000072196126 0.23123123683035374
```

### 1.3 文件放置规范

文件树如下

![](./doc/File_tree.png)

### 1.4 选择一个您需要的模型

在文件夹 `./models` 下选择一个你需要的模型然后复制一份出来，将文件开头的 `nc = ` 修改为数据集的分类数，下面是借鉴 `./models/yolov5m.yaml`来修改的

```yaml
# parameters
nc: 9 # number of classes     # <============ 修改这里为数据集的分类数
depth_multiple: 0.67 # model depth multiple
width_multiple: 0.75 # layer channel multiple

# anchors
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Focus, [64, 3]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, BottleneckCSP, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 9, BottleneckCSP, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, BottleneckCSP, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 1, SPP, [1024, [5, 9, 13]]],
    [-1, 3, BottleneckCSP, [1024, False]], # 9
  ]

# YOLOv5 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, BottleneckCSP, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, BottleneckCSP, [256, False]], # 17

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, BottleneckCSP, [512, False]], # 20

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, BottleneckCSP, [1024, False]], # 23

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```

### 1.5 开始训练

这里选择了 `yolov5m` 模型进行训练，权重也是基于 `yolov5m.pt` 来训练

```shell script
#!/usr/bin/env sh
echo "train begin..."
id=1231
epochs=100
data='data/custom_data.yaml'
batch_size=16
data_dir=/home/zq/work/data/underground_mine/
weights=/home/zq/work/test/Smart_Construction-master/weights/1227/weights/best.pt
rm ${data_dir}*/*/*.cache   # 删除缓存  防止添加数据后仍训练缓存中旧的数据
nohup python train.py --epochs ${epochs} --data ${data} --weights ${weights} --batch-size ${batch_size}> ${id}.log 2>&1 &
```

其中，`yolov5m.pt` 需要自行下载放在本工程的根目录即可，下载地址 [官方权重](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)

### 1.6 看训练之后的结果

训练之后，权重会保存在 `./runs` 文件夹里面的每个 `exp` 文件里面的 `weights/best.bt` ，里面还可以看到训练的效果

# 2. 侦测

侦测图片会保存在 `./inferenct/output/` 文件夹下

运行命令：

```shell script
python detect.py --source   0  # webcam
                            file.jpg  # image
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

# 3. 生成 ONNX

## 3.1 安装 `onnx` 库

```shell script
pip install onnx
```

## 3.2 执行生成

```shell script
python ./models/export.py --weights ./weights/helmet_head_person_s.pt --img 640 --batch 1
```

`onnx` 和 `torchscript` 文件会生成在 `./weights` 文件夹中

# 4. 增加数据集的分类

关于增加数据集分类的方法：

`SHWD` 数据集里面的 `person` 指的是`头（head）`，没有 `人体` 的类别，先将现有的自己的数据集执行脚本生成 yolov5 需要的标签文件 `.txt`，之后再用 `yolov5x.pt` 加上 `yolov5x.yaml` ，使用预训练的模型检测出人体，保存人体的坐标信息并转换成`yolov5`所需的`txt`格式。

```shell script
python detect.py --save-txt --source 自己数据集的文件目录 --weights ./weights/yolov5x.pt
```

`yolov5` 会推理出所有的分类，并在 `inference/output` 中生成对应图片的 `.txt` 标签文件；

修改 `./data/gen_data/merge_data.py` 中的自己数据集标签所在的路径，执行这个 python 脚本，会进行 `人体(person)` 类型的合并

---

**如果帮到您请给个 star ✨✨✨，您的 star 是我最大的鼓励！**

**如果能帮到您的项目快速落地，可以 buy me a coffee ☕**

![](./doc/微信图片_20220105103117.jpg)
