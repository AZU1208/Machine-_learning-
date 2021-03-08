# openCVで画像をグレーにする。
# 画像認識を簡単にするためにグレーにする。

import cv2

img = cv2.imread(元画像のパス, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(グレースケール画像のパス, img)



# ヒストグラムを描画

import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_hist(img):
    img_hist = np.histogram(img.ravel(), 256, [0, 256])
    hist = img_hist[0]
    plt.bar(np.arange(256), hist)
    plt.show()

plot_hist(cv2.imread(ヒストグラムを表示する画像のパス, cv2.IMREAD_GRAYSCALE))
