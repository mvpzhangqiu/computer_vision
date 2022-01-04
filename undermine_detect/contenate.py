'''
抠图，拼底图
'''

import cv2
import os
import glob
import json
import numpy as np

img_bg_filenames = glob.glob('/home/zq/work/data/underground_mine/helmet/bg/*.jpg')

img_src_dir = '/home/zq/work/data/underground_mine/helmet/source/'
# img_bg_filenames = glob.glob('data/bg/*')

# for img_bg_filename in glob.glob('/home/zq/work/data/underground_mine/helmet/bg/*.jpg'):
#     img_bg = cv2.imread(img_bg_filename)
#     cv2.imwrite(img_bg_filename, cv2.resize(img_bg, (1920, 1084), interpolation=cv2.INTER_NEAREST))

with open('/home/zq/work/data/underground_mine/helmet/fenge.json') as load_f:
    contents = json.load(load_f)
    i = 0
    for content in contents:
        img_src_filename = os.path.join(img_src_dir, contents[content]['filename'])
        img_src = cv2.imread(img_src_filename)
        # mask = np.zeros(img_src.shape[:2])
        mask_src = np.zeros_like(img_src)
        mask_bg = np.ones_like(img_src)
        regions = np.array(contents[content]['regions'])
        for region in regions:
            all_points_x = region['shape_attributes']['all_points_x']
            all_points_y = region['shape_attributes']['all_points_y']
            pts = list(zip(all_points_x, all_points_y))
            pts = np.array(pts)
            cv2.fillPoly(mask_src, [pts], (1, 1, 1))
        for img_bg_filename in img_bg_filenames:
            img_bg = cv2.imread(img_bg_filename)
            img_merge = img_src * mask_src + img_bg * (mask_bg - mask_src)
            cv2.imwrite('/home/zq/work/data/underground_mine/helmet/merge/{}_{}.jpg'.format(os.path.basename(img_bg_filename).split('.')[0], i), img_merge)
            i += 1

