#-*- coding:utf-8 ?*-
#author:yuanly
#data:2021.4.6
#Description:json?????????yolo??????
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

def rm_depulicate(txt_path):
    txts = getAllPath(txt_path, ".txt")
    for txt in txts:
        with open(txt) as fp1:
            lines = fp1.readlines()
            lines = list(set(lines))
        fp1.close()
        if len(lines) == 0:
            continue
        with open(txt, mode='w') as fp2:
            for i, line in enumerate(lines):
                if i == 0:
                    fp2.write(line.strip())
                else:
                    fp2.write('\n' + line.strip())
        fp2.close()


def main(img_dir,json_dir,txt_dir):
    json_dirs = getAllPath(json_dir, '.json')
    for json_dir in json_dirs:
        # 读取 json 文件数据
        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)
        # 循环处理
        for t in content:
            tmp = content[t]['filename'].split('.')
            # filename = txt_dir + content[t]['filename'].replace("jpg", "txt")
            filename = txt_dir + tmp[0] + '.txt'
            img =cv2.imread(img_dir + content[t]['filename'])
            if img is None:
                continue
            dh = 1./img.shape[0]
            dw = 1./img.shape[1]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)

            fp = open(filename, mode="w+", encoding="utf-8")
            # with open(filename, mode="w+", encoding="utf-8") as fp:
            object = content[t]['regions']
            for t in object:
                # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值,原图的width，height归一化
                width =t['shape_attributes']['width']
                height =t['shape_attributes']['height']

                x = (t['shape_attributes']['x'] + t['shape_attributes']['width'] / 2)*dw
                y = (t['shape_attributes']['y'] + t['shape_attributes']['height'] / 2)*dh
                w = width*dw
                h = height*dh
                if t['region_attributes']['categories']=='20':
                    #蒙版处理
                    continue
                file_str = t['region_attributes']['categories'] + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                           ' ' + str(round(h, 6))+'\n'
                print(file_str)
                line_data = fp.readlines()

                if len(line_data) != 0:
                    fp.write('\n' + file_str)
                else:
                    fp.write(file_str)
            fp.close()


def main_flir(img_dir,json_dir,txt_dir):
    json_dirs = getAllPath(json_dir, '.json')
    for json_dir in json_dirs:
        with open(json_dir, 'r') as load_f:
            content = json.load(load_f)
        
        for t in content["annotations"]:
            # filename = content["images"][t["image_id"]]["file_name"].split("/")[1].split('.')[0] + '.txt'
            tmp = content[t]['filename'].split('.')
            filename = txt_dir + tmp[0] + '.txt'
            img = cv2.imread(os.path.join(img_dir, filename.replace("txt", "jpeg")))
            if img is None:
                continue
            width = img.shape[0]
            height = img.shape[1]
            dh = 1./img.shape[0]
            dw = 1./img.shape[1]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)
            
            fp = open(os.path.join(txt_dir, filename), mode="a+", encoding="utf-8")
            x = (t["bbox"][0] + t["bbox"][2] / 2)*dw
            y = (t["bbox"][1] + t["bbox"][3] / 2)*dh
            w = t["bbox"][2]*dw
            h = t["bbox"][3]*dh
            
            file_str = str(t['category_id']) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                           ' ' + str(round(h, 6))+'\n'
            line_data = fp.readlines()

            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)
            fp.close()

if __name__ == '__main__':
    # json_dir = '/home/zq/work/data/cnl_undetected/'
    # txt_dir = '/home/zq/work/data/labels/train/cnl_undetected/'
    # img_dir = '/home/zq/work/data/images/train/cnl_undetected/'

    json_dir = '/home/zq/work/data/underground_mine/zoudao/'
    txt_dir = '/home/zq/work/data/underground_mine/zoudao/txt_total/'
    img_dir = '/home/zq/work/data/underground_mine/zoudao/img_total/'

    main(img_dir, json_dir, txt_dir)
    print("11")
