import glob
import os
import cv2
import numpy as np
import skimage.feature
import skimage.segmentation

# class names
# names: ['person', 'head', 'helmet']
# "0":"helmet_yes","1":"helmet_no","2":"person",

# txt_dir = '/home/zq/work/data/underground_mine/YM_1-5900/txt_total/'
# txt_filenames = glob.glob('/home/zq/work/data/underground_mine/YM_1-5900/txt_total/*.txt')
# for txt_filename in txt_filenames:
#     with open(txt_filename, "r") as f:
#         data = f.readlines()
#         for line in data:
#             if line.split()[0] == '1':
#                 print(txt_filename)


# txt_dir = '/home/zq/work/data/underground_mine/Safety_Helmet_Train_dataset/score/labels/total/'
# txt_target_dir = '/home/zq/work/data/underground_mine/Safety_Helmet_Train_dataset/score/labels/total_tran/'
#
# if not os.path.exists(txt_target_dir):
#     os.mkdir(txt_target_dir)
#
# txt_filenames = glob.glob(txt_dir + '*.txt')
# for txt_filename in txt_filenames:
#     with open(txt_filename, "r") as f:
#         data = f.readlines()
#         l = []
#         for line in data:
#             if line.split()[0] == '0':  # person
#                 line = line.replace('0', '2', 1)
#             elif line.split()[0] == '2':  # helmet
#                 line = line.replace('2', '0', 1)
#             l.append(line)
#         with open(txt_target_dir + os.path.basename(txt_filename), 'w') as fw:
#             fw.writelines(l)


# with open('/home/zq/work/data/underground_mine/Safety_Helmet_Train_dataset/score/000011.txt', "r") as f:
#     data = f.readlines()
#     print(data)
#     l = []
#     for line in data:
#         if line.split()[0] == '0':  # person
#             line = line.replace('0', '2', 1)
#         elif line.split()[0] == '1':  # head
#             line = line.replace('1', '1', 1)
#         elif line.split()[0] == '2':  # helmet
#             line = line.replace('2', '0', 1)
#         l.append(line)
#     with open('/home/zq/work/data/underground_mine/Safety_Helmet_Train_dataset/score/000012.txt', "w") as fw:
#         fw.writelines(l)


# img_liang = cv2.imread('data/1.jpg')
# img_an = cv2.imread('data/2.jpg')




"""
注释的内容为灰度世界假设算法
"""
# for i in range(m):
#     for j in range(n):
#         if(sum[i][j])
# sum_b, sum_g, sum_r = np.sum(np.ravel(b)), np.sum(np.ravel(g)), np.sum(np.ravel(r))
# avl_b, avl_g, avl_r = sum_b / (m * n), sum_g / (m * n), sum_r / (m * n)
# gray=(avl_b + avl_r + avl_g) / 3
# k_r , k_g , k_b = gray / avl_r , gray / avl_g , gray / avl_b
# for i in range(m):
#     for j in range(n):
#         b[i][j]=b[i][j] * k_b
#         g[i][j]=g[i][j] * k_g
#         r[i][j]=r[i][j] * k_r
# img_0 = cv2.merge([b,g,r])

# cv2.imshow('修图',img_0)




# img_bilater_liang = cv2.bilateralFilter(img_liang,9,75,75)
# img_bilater_an = cv2.bilateralFilter(img_an,9,75,75)
# cv2.imwrite('data/img_bilater_liang.jpg', img_bilater_liang)
# cv2.imwrite('data/img_bilater_an.jpg', img_bilater_an)

# img_liang_hsv = cv2.cvtColor(img_liang, cv2.COLOR_BGR2HSV)
# img_an_hsv = cv2.cvtColor(img_an, cv2.COLOR_BGR2HSV)
#
# img_liang_hsv[:, :, 2] = 128
# img_an_hsv[:, :, 2] = 128
#
# img_liang_hsv = cv2.cvtColor(img_liang_hsv, cv2.COLOR_HSV2BGR)
# img_an_hsv = cv2.cvtColor(img_an_hsv, cv2.COLOR_HSV2BGR)




# cv2.imwrite('data/1_hsv.jpg', img_liang_hsv)
# cv2.imwrite('data/2_hsv.jpg', img_an_hsv)


img_liang = cv2.imread('data/1.jpg')
img_an = cv2.imread('data/2.jpg')

gray_liang = cv2.cvtColor(img_liang, cv2.COLOR_BGR2GRAY)
gray_an = cv2.cvtColor(img_an, cv2.COLOR_BGR2GRAY)

lbp_liang = skimage.feature.local_binary_pattern(gray_liang, 899, 30.0, method='default')
lbp_an = skimage.feature.local_binary_pattern(gray_an, 899, 30.0, method='default')

cv2.imwrite('data/lbp_liang.jpg', lbp_liang)
cv2.imwrite('data/lbp_an.jpg', lbp_an)
cv2.imwrite('data/diff.jpg', lbp_liang - lbp_an)
