import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time

# 验证码图片的存放路径
CAPTCHA_IMAGE_PATH = '/home/yanwu9887/PycharmProjects/project_1/captcha/images/'
# 验证码图片的宽度
CAPTCHA_IMAGE_WIDHT = 160
# 验证码图片的高度
CAPTCHA_IMAGE_HEIGHT = 60

CHAR_SET_LEN = 10
CAPTCHA_LEN = 4

# 60%的验证码图片放入训练集中
TRAIN_IMAGE_PERCENT = 0.6
# 训练集，用于训练的验证码图片的文件名
TRAINING_IMAGE_NAME = []
# 验证集，用于模型验证的验证码图片的文件名
VALIDATION_IMAGE_NAME = []
#存放训练好的模型的路径
MODEL_SAVE_PATH = '/home/yanwu9887/PycharmProjects/project_1/captcha/models/'

def get_image_file_name(imgPath=CAPTCHA_IMAGE_PATH):
    fileName = []
    total = 0
    for filePath in os.listdir(imgPath):
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        total += 1
    return fileName, total

#将验证码转换为训练时用的标签向量，维数是 40
#例如，如果验证码是 ‘0296’ ，则对应的标签是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def name2label(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i*CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label

#取得验证码图片的数据以及它的标签
def get_data_and_label(fileName, filePath=CAPTCHA_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten()/255
    image_label = name2label(fileName[0:CAPTCHA_LEN])
    return image_data, image_label

# 生成一个训练batch
def get_next_batch(batchSize=32, trainOrTest='train', step=0):
    batch_data = np.zeros([batchSize, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])
    fileNameList = TRAINING_IMAGE_NAME
    if trainOrTest == 'validate':
        fileNameList = VALIDATION_IMAGE_NAME

    totalNumber = len(fileNameList)
    indexStart = step * batchSize
    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]
        img_data, img_label = get_data_and_label(name)
        batch_data[i, :] = img_data
        batch_label[i, :] = img_label

    return batch_data, batch_label
