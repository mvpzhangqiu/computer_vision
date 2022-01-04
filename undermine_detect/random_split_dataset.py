import os, random, shutil


def moveFile(fileDir_txt, fileDir_img, tarDir_txt, tarDir_img):
    pathDir = os.listdir(fileDir_txt)  # 取txt的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        txt_name = name
        img_name = name.replace('txt', 'jpg')
        shutil.move(fileDir_txt + txt_name, tarDir_txt + txt_name)
        shutil.move(fileDir_img + img_name, tarDir_img + img_name)
    return


if __name__ == '__main__':
    fileDir_txt = "/home/zq/work/data/underground_mine/zoudao/txt_train/"  # 源txt文件夹路径
    fileDir_img = "/home/zq/work/data/underground_mine/zoudao/img_train/"  # 源图片文件夹路径
    tarDir_txt = '/home/zq/work/data/underground_mine/zoudao/txt_val/'  # 移动到新的txt文件夹路径
    tarDir_img = '/home/zq/work/data/underground_mine/zoudao/img_val/'  # 移动到新的图片文件夹路径
    os.makedirs(tarDir_txt, exist_ok=True)
    os.makedirs(tarDir_img, exist_ok=True)
    moveFile(fileDir_txt, fileDir_img, tarDir_txt, tarDir_img)
