import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

CAPTCHA_LEN = 4

MODEL_SAVE_PATH = '/home/yanwu9887/PycharmProjects/project_1/captcha/models/'
TEST_IMAGE_PATH = '/home/yanwu9887/PycharmProjects/project_1/captcha/test/'


def get_image_data_and_name(fileName, filePath=TEST_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    # 转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten() / 255
    image_name = fileName[0:CAPTCHA_LEN]
    return image_data, image_name

