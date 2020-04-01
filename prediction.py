# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import load_model

model_path = 'F:/Project/traning_model.h5'
model = load_model(model_path)

IMG_SIZE = 150

def labelling(result):
  answer =0
  for i in range(result):
    for j in range(result):
      if(result[i][j]==1):
        #print(answer)
        if answer == 0:
          print("ru")
        elif answer == 1:
          print("a")
        elif answer == 2:
          print("Aa")
        elif answer == 3:
          print("i")
        elif answer == 4:
          print("I")
        elif answer == 5:
          print("u")
        elif answer == 6:
          print("U")
        elif answer == 7:
          print("e")
        elif answer == 8:
          print("ai")
        elif answer == 9:
          print("o")
        elif answer == 10:
          print("au")
        elif answer == 11:
          print("am")
        elif answer == 12:
          print("ah")
        elif answer == 13:
          print("ka")
        elif answer == 14:
          print("kha")
        elif answer == 15:
          print("g")
        elif answer == 16:
          print("gh")
        elif answer == 17:
          print("ch")
        elif answer == 18:
          print("chh")
        elif answer == 19:
          print("j")
        elif answer == 20:
          print("jh")
        elif answer == 21:
          print("T")
        elif answer == 22:
          print("Th")
        elif answer == 23:
          print("D")
        elif answer == 24:
          print("Dh")
        elif answer == 25:
          print("N")
        elif answer == 27:
          print("th")
        elif answer == 28:
          print("d")
        elif answer == 29:
          print("dh")
        elif answer == 30:
          print("n")
        elif answer == 31:
          print("p")
        elif answer == 32:
          print("ph")
        elif answer == 33:
          print("b")
        elif answer == 34:
          print("bh")
        elif answer == 35:
          print("m")
        elif answer == 36:
          print("y")
        elif answer == 37:
          print("r")
        elif answer == 38:
          print("l")
        elif answer == 39:
          print("v")
        elif answer == 40:
          print("sh")
        elif answer == 41:
          print("SH")
        elif answer == 42:
          print("s")
        elif answer == 43:
          print("h")
        elif answer == 44:
          print("al")
        elif answer == 45:
          print("ksh")
        elif answer == 46:
          print("gy")
        else:
          print("Other")
      answer += 1

prediction_list = []
img_loc = 'F:/Project/images/predictit.png'
try:
    img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    predict_this = np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    plt.imshow(img, cmap='gray')
    array = model.predict(predict_this)
    labelling(array)
except:
    print("Error at :"+ img_loc)