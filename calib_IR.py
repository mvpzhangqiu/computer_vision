# -*- coding: utf-8 -*-
"""
Picture File Folder: ".\pic\IR_camera_calib_img", With Distort. 
By mvpzhangqiu
"""

import os
import numpy as np
import cv2
import glob


def calib(inter_corner_shape, size_per_grid, img_dir, img_type):
    # inter_corner_shape 角点shape
    # size_per_grid 每个网格的边长（世界坐标）
    w, h = inter_corner_shape
    # 角点网格坐标 (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h, 3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)  # np.mgrid 生成等差数组
    # cp_world: 角点网格坐标转换为世界坐标
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    # 图片灰度化后通过cv2.findChessboardCorners寻找角点坐标
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the corners, cp_img: corner points in pixel space.
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h), None)
        # if ret is True, save.
        if ret == True:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # view the corners
            # cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            # cv2.imshow('FoundCorners',img)
            # cv2.waitKey(1)
    #cv2.destroyAllWindows()
    
    # calibrate the camera 相机标定
    # cv2.calibrateCamera
    '''
    参数：
    obj_points：世界坐标点，img_points：像素坐标点
    返回值：
    mat_inter：内参矩阵，coff_dis：畸变矩阵，v_rot：旋转向量，v_trans：位移向量
    '''
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    print (("ret:"),ret)
    print (("internal matrix:\n"),mat_inter)
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print (("distortion cofficients:\n"),coff_dis)  
    print (("rotation vectors:\n"),v_rot)
    print (("translation vectors:\n"),v_trans)
    # calculate the error of reproject 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        # 函数cv2.projectPoints通过给定的内参和外参计算三维点投影到二维图像平面上的坐标
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        # 计算L2误差
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        total_error += error
    print(("Average Error of Reproject: "), total_error/len(obj_points))
    return mat_inter, coff_dis
    
def dedistortion(inter_corner_shape, img_dir,img_type, save_dir, mat_inter, coff_dis):
    w,h = inter_corner_shape
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        # cv2.getOptimalNewCameraMatrix 调节视场大小，为1时视场大小不变，小于1时缩放视场。
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter,coff_dis,(w,h),0,(w,h)) # 自由比例参数
        '''
        cv2.undistort利用求得的相机的内参和外参数据，对图像进行畸变的矫正
        参数：
        img：畸变的原始图像，mat_inter：内参矩阵，coff_dis：畸变系数
        '''
        dst = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)
        # clip the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(save_dir + os.sep + img_name, dst)
    print('Dedistorted images have been saved to: %s successfully.' %save_dir)
    
if __name__ == '__main__':
    inter_corner_shape = (11,8)
    size_per_grid = 0.02
    img_dir = ".\\pic\\IR_camera_calib_img"
    img_type = "png"
    # calibrate the camera
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir,img_type)
    # dedistort and save the dedistortion result. 
    save_dir = ".\\pic\\save_dedistortion"
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis)
    
    
    
