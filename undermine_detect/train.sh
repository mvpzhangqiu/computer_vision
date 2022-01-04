#!/usr/bin/env sh
#nohup python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --weights yolov5m.pt >> log0729.txt
#python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --cfg models/yolov5s.yaml

echo "train begin..."
id=1231
epochs=10
#data='data/scraper.yaml'
data='data/custom_data.yaml'
batch_size=12
data_dir=/home/zq/work/data/underground_mine/
weights=/home/zq/work/test/Smart_Construction-master/weights/1227/weights/best.pt
rm ${data_dir}*/*/*.cache
nohup python train.py --epochs ${epochs} --data ${data} --weights ${weights} --batch-size ${batch_size}> ${id}.log 2>&1 &


