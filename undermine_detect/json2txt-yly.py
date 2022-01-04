#-*- coding:utf-8 –*-
#author:yuanly
#data:2021.4.6
#Description:json格式标注文件转换为yolo数据标注格式
import os
import json
import cv2

# 获取路径的所有指定后缀文件
def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray
def main(img_dir,json_dir,txt_dir,img_labels_dir):
    json_dirs = getAllPath(json_dir, '.json')
    for json_dir in json_dirs:
        # 读取 json 文件数据
        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)
        # 循环处理
        for t in content:
            tmp = content[t]['filename'].split('.')
            filename = txt_dir + tmp[0] + '.txt'
            img =cv2.imread(img_dir + content[t]['filename'])
            if img is None:
                continue
            print(tmp)
            dh = 1./img.shape[0]
            dw = 1./img.shape[1]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)
            fp = open(filename, mode="w+", encoding="utf-8")
            object = content[t]['regions']
            for t in object:
                # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值,原图的width，height归一化
                width =t['shape_attributes']['width']
                height =t['shape_attributes']['height']
                # cv2.rectangle(img, (t['shape_attributes']['x'], t['shape_attributes']['y']),
                #               (t['shape_attributes']['x']+width, t['shape_attributes']['y']  + height), (0, 255, 0), 2)

                x = (t['shape_attributes']['x'] + t['shape_attributes']['width'] / 2)*dw
                y = (t['shape_attributes']['y'] + t['shape_attributes']['height'] / 2)*dh
                w = width*dw
                h = height*dh
                #fp = open(filename, mode="r+", encoding="utf-8")
                if t['region_attributes']['categories']=='20':
                    #蒙版处理
                    continue
                print (x)
                file_str = t['region_attributes']['categories'] + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                           ' ' + str(round(h, 6))+'\n'
                line_data = fp.readlines()

                if len(line_data) != 0:
                    fp.write('\n' + file_str)
                else:
                    fp.write(file_str)
            fp.close()
            # cv2.imwrite(img_labels_dir + tmp[0]+'.jpg', img)
#
def main_flir(img_dir, json_dir, txt_dir):
    json_dirs = getAllPath(json_dir, '.json')
    l = []
    for json_dir in json_dirs:
        # ?? json ????
        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)
        for t in content["annotations"]:
            filename = content["images"][t["image_id"]]["file_name"].split("/")[1].split('.')[0] + '.txt'
            img = cv2.imread(os.path.join(img_dir, filename.replace("txt", "jpeg")))
            if img is None:
                continue
            dh = 1. / img.shape[0]
            dw = 1. / img.shape[1]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)
            print(os.path.join(txt_dir, filename))

            fp = open(os.path.join(txt_dir, filename), mode="a+", encoding="utf-8")
            x = (t["bbox"][0] + t["bbox"][2] / 2) * dw
            y = (t["bbox"][1] + t["bbox"][3] / 2) * dh
            w = t["bbox"][2] * dw
            h = t["bbox"][3] * dh
            # print(x, y, w, h)
            file_str = str(t['category_id']) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
                round(w, 6)) + \
                       ' ' + str(round(h, 6)) + '\n'
            line_data = fp.readlines()

            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)
            fp.close()

if __name__ == '__main__':
    json_dir = '/home/zq/work/data/underground_mine/meibi/json/'
    txt_dir = '/home/zq/work/data/underground_mine/meibi/txt_total/'
    img_dir = '/home/zq/work/data/underground_mine/meibi/img_total/'
    img_labels_dir = '/home/yly/work/dataset/suizhong_night/images_labels/'  # 图片路径读取信息获取图像宽高
    main(img_dir,json_dir,txt_dir,img_labels_dir)