'''
@Author: your name
@Date: 2020-07-21 21:01:03
LastEditTime: 2020-09-23 09:02:19
LastEditors: Liu Chen
@Description: preprocessing 该文件用于制作 现实主义数据集
        步骤：1，运行crop_dir函数，把原始图像按照中心裁剪的方式保留正方形。
            2，运行main函数，设置最终的输出目录 base_root。
                注意：a) 所有的crop_dir里面的图像文件都需要用数字命名且不可命名重复
                      b) 注意修改目录 crop_dir和base_root 为你的目录路径
FilePath: \GatherAlgorithms\deep_learning\seq_models\scripts\work0721.py
@  
'''


import cv2
import shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import json
import math
from math import *
from os.path import join as opj
from tqdm import tqdm


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def crop_center_save(fp, size, savepath):
    """
        裁切中心并且保存
        fp: filepath (abs)
        size: tuple (w,h) 
        savepath: dir (abs)
    """
    # read in image
    # img = cv_imread(fp)
    # img = plt.imread(fp)
    img = Image.open(fp)
    fname = os.path.split(fp)[1]
    num = int(fname[4:].split('.')[0])-33
    (w, h) = img.size
    center = (w//2, h//2)
    luw = center[0]-size[0]//2
    luh = center[1] - size[1]//2
    rdw = center[0] + size[0] // 2
    rdh = center[1] + size[1] // 2
    crop_img = img.crop([luw, luh, rdw, rdh])
    # plt.imshow(crop_img)
    crop_img.save(opj(savepath, str(num)+'.jpg'))
    # plt.show()


def crop_dir():
    """
    crop all images in a directory and save them in another directory
    """
    fp = r'J:\research\datasets\NWPU-ChangAn\NWPU-ChangAn'
    savepath = r'J:\research\datasets\NWPU-ChangAn\NWPU-ChangAn_Rect'

    for name in tqdm(os.listdir(fp)):
        file = opj(fp, name)
        crop_center_save(file, (2000, 2000), savepath)


def read_json(file):
    """
        read json file
    """
    f = json.load(open(file, 'r', encoding='utf-8'))
    return f['shapes']


def get_corners(leftup, rightdown):
    pt = [[0, 0] for _ in range(4)]
    pt[0] = leftup
    pt[2] = rightdown
    pt[1][0] = (pt[0][0]+pt[2][0] + (pt[2][1]-pt[0][1]))/2
    pt[1][1] = (pt[0][1]+pt[2][1] + (pt[0][0]-pt[2][0]))/2
    pt[3][0] = (pt[0][0]+pt[2][0] - (pt[2][1]-pt[0][1]))/2
    pt[3][1] = (pt[0][1]+pt[2][1] - (pt[0][0]-pt[2][0]))/2
    return pt


def cal_coors_afterrotate_v2(point, center, imgsize, theta):
    col, row = imgsize
    x1 = point[0]
    y1 = row - point[1]
    x2 = center[0]
    y2 = row - center[1]
    x = (x1 - x2)*math.cos(math.pi / 180.0 * theta) - \
        (y1 - y2)*math.sin(math.pi / 180.0 * theta) + x2
    y = (x1 - x2)*math.sin(math.pi / 180.0 * theta) + \
        (y1 - y2)*math.cos(math.pi / 180.0 * theta) + y2
    x = int(x)
    y = int(row - y)
    return [x, y]


def cal_coors_afterrotate(point, rotateMat):
    p = [0, 0]
    [[p[0]], [p[1]]] = np.dot(
        rotateMat, np.array([[point[0]], [point[1]], [1]]))
    return [int(p[0]), int(p[1])]


def crop_satellite(img_file, json_file, save_path, info_path):
    """读入图像和标注json文件，把图像裁剪出来矩形"""
    shape_infos = read_json(json_file)

    # 读入图像
    sateimg = cv_imread(img_file)
    (w, h, c) = sateimg.shape

    for p in shape_infos:
        fname = p['label']+'.jpg'
        ps = p['points']
        pt = [(0, 0) for _ in range(4)]  # corner point after rotation
        # 点ps描述的内容是正方形的左上角点和右下角点
        # 现在需要根据ps，从sateimg中截下这一个正方形
        # 思路是先计算出要把satelite旋转的角度，旋转后再根据新的ps获取
        line = math.sqrt((ps[0][0]-ps[1][0])**2 + (ps[0][1]-ps[1][1])**2)
        height = ps[0][0]-ps[1][0]
        ang = math.asin(height/line) * (180/math.pi)

        if ps[0][1] < ps[1][1]:
            angle = -45-ang  # 第2个点（终止点）为原点 -45 ~ 90度范围内为逆时针旋转
        else:
            angle = 45-(-90-ang)  # 其余情况都是顺时针旋转角度
        # print(ang,angle, ps)
        center = ((ps[0][0]+ps[1][0])//2, (ps[0][1]+ps[1][1])//2)
        # 获得正方形的四个角的坐标，顺序（左上，右上，右下，左下）

        heightNew = int(w * fabs(sin(radians(angle))) +
                        h * fabs(cos(radians(angle))))
        widthNew = int(h * fabs(sin(radians(angle))) +
                       w * fabs(cos(radians(angle))))

        rotateMat = cv2.getRotationMatrix2D(
            center, -angle, 1)  # 其中旋转角度，负数为顺时针旋转，和习惯刚好反过来
        print(angle)
        imgRotation = cv2.warpAffine(
            sateimg, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
        # 获取四点坐标以及旋转后的四点坐标
        pt = get_corners(ps[0], ps[1])
        for i in range(len(pt)):
            # pt[i] = cal_coors_afterrotate_v2(pt[i],center,(w,h),angle)
            pt[i] = cal_coors_afterrotate(pt[i], rotateMat)

        # 处理反转的情况
        if pt[0][0] > pt[1][0]:
            pt[0][0], pt[1][0] = pt[1][0], pt[0][0]
        if pt[0][1] > pt[3][1]:
            pt[0][1], pt[3][1] = pt[3][1], pt[0][1]

        imgOut = imgRotation[int(pt[0][1]):int(
            pt[3][1]), int(pt[0][0]):int(pt[1][0])]
        # cv2.imwrite(opj(save_path, fname), imgOut)
        foldername = fname.split('.')[0]
        produce_datagroup(imgRotation, pt, opj(
            save_path, foldername), info_path)
        print(pt)

# def generate_new():
#     skip_is_in


def produce_datagroup(source_img, center_patch, save_path, info_path):
    '''
    输入源图像，中心点坐标，生成一个文件夹的周遭图片块们
    @description: input sourcefile and point coords, 
        output a folder with all patches around the centerpatch 
    @param {type} :center_patch:[(x1,y1),(x2,y2)]
                    info_path:保存信息的文件夹，通常设在savepath的外层
    @return: None
    '''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(info_path):
        os.mkdir(info_path)
    updown = [center_patch[0][1], center_patch[3][1]]
    leftright = [center_patch[0][0], center_patch[1][0]]

    shift_scales = [i for i in range(-500, 500, 40)]
    # shift_scales.remove(0)
    offsets = [0] + shift_scales

    fname = 0
    with open(opj(info_path, os.path.split(save_path)[1]), 'w') as fp:
        for v_shift in offsets:
            for h_shift in offsets:
                imgOut = source_img[updown[0]+v_shift:updown[1]+v_shift,
                                    leftright[0]+h_shift:leftright[1]+h_shift]
                cv2.imwrite(opj(save_path, str(v_shift) +
                                '_'+str(h_shift)+'.jpg'), imgOut)
                fname += 1
                cent_coord_y = (updown[0]+v_shift+updown[1]+v_shift)//2
                cent_coord_x = (leftright[0]+h_shift+leftright[1]+h_shift)//2
                fp.write(str(v_shift)+'_'+str(h_shift)+' ' +
                         str(cent_coord_x) + ' ' + str(cent_coord_y) + '\n')


def doublecheck(save_path):
    """
        对于该文件夹下的所有子文件夹内的图片进行检查。如果发现空文件或者非矩形文件，则去除
    """
    for folder in tqdm(os.listdir(save_path)):
        img_dir = opj(save_path, folder)
        for img in os.listdir(img_dir):
            imgfile = opj(img_dir, img)
            try:
                cont = cv_imread(imgfile)
            except:
                os.remove(imgfile)
                print('Empty image {} {}'.format(folder, img))
                continue
            (w, h, c) = cont.shape
            if abs(w - h) > 8:  # 切割后长宽相差太大的不要
                os.remove(imgfile)
        # 如果处理以后该文件夹内一张图片都没有，就把这个文件夹删除
        if len(os.listdir(img_dir)) == 0:
            os.removedirs(img_dir)


def main():
    # NWPU-ChangAn 数据
    # base_root = r'J:\work_202007_10\data'
    # sp = opj(base_root, r'20200722_2.tif')
    # jsonfiles =  ['20200722_2.json', '50_109.json', '110_169.json']

    base_root = r'J:\work_202007_10\data_DaMing'
    sp = opj(base_root, '20200825.tif')
    # 对于多个不同的人标注的json文件，分别处理，自动合并
    jsonfiles = ['DaMing_Palace_chenwang.json', 'DaMing_Palace_lichao.json']

    save = opj(base_root, 'crop')
    info = opj(base_root, 'cropdb')
    if not os.path.exists(save):
        os.mkdir(save)
    if not os.path.exists(info):
        os.mkdir(info)
    for fname in jsonfiles:
        jp = opj(base_root, fname)
        crop_satellite(sp, jp, save, info)

    doublecheck(save)


if __name__ == '__main__':
    main()
