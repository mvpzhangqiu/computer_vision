#!/usr/bin/env sh

echo "test begin..."
id=1214
weights='/home/zq/work/test/Smart_Construction-master/weights/1217/weights/best.pt'  # 训练集包含公开安全帽数据集
data='data/test.yaml'

nohup python test.py  --weights ${weights} --data ${data} --verbose > 1214.log 2>&1 &