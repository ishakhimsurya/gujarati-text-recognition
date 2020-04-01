from flask import render_template, request, jsonify, Flask, redirect
import flask
import numpy as np
import traceback
import pickle
import cv2
import os
from PIL import Image

IMG_SIZE = 150

def labelling(result):
      #print(result)
  for i in range(result.shape[0]):
    #print(i)
    answer = 0
    for j in range(result[i].shape[0]):
      if(result[i][j]==1):
        #print(answer)
        if answer == 0:
          return("ru")
        elif answer == 1:
          return("a")
        elif answer == 2:
          return("Aa")
        elif answer == 3:
          return("i")
        elif answer == 4:
          return("I")
        elif answer == 5:
          return("u")
        elif answer == 6:
          return("U")
        elif answer == 7:
          return("e")
        elif answer == 8:
          return("ai")
        elif answer == 9:
          return("o")
        elif answer == 10:
          return("au")
        elif answer == 11:
          return("am")
        elif answer == 12:
          return("ah")
        elif answer == 13:
          return("ka")
        elif answer == 14:
          return("kha")
        elif answer == 15:
          return("g")
        elif answer == 16:
          return("gh")
        elif answer == 17:
          return("ch")
        elif answer == 18:
          return("chh")
        elif answer == 19:
          return("j")
        elif answer == 20:
          return("jh")
        elif answer == 21:
          return("T")
        elif answer == 22:
          return("Th")
        elif answer == 23:
          return("D")
        elif answer == 24:
          return("Dh")
        elif answer == 25:
          return("N")
        elif answer == 27:
          return("th")
        elif answer == 28:
          return("d")
        elif answer == 29:
          return("dh")
        elif answer == 30:
          return("n")
        elif answer == 31:
          return("p")
        elif answer == 32:
          return("ph")
        elif answer == 33:
          return("b")
        elif answer == 34:
          return("bh")
        elif answer == 35:
          return("m")
        elif answer == 36:
          return("y")
        elif answer == 37:
          return("r")
        elif answer == 38:
          return("l")
        elif answer == 39:
          return("v")
        elif answer == 40:
          return("sh")
        elif answer == 41:
          return("SH")
        elif answer == 42:
          return("s")
        elif answer == 43:
          return("h")
        elif answer == 44:
          return("al")
        elif answer == 45:
          return("ksh")
        elif answer == 46:
          return("gy")
        else:
          return("Other")
      answer += 1

# App definition
app = Flask(__name__, template_folder='templates')
app.config["IMAGE_UPLOADS"] = "G:/SGP/gujarati-text-recognition/api/media/uploads/"#
# app.config["IMAGE_UPLOADS"] = "path_to_upload"


# importing models
with open('model/finalized_model.sav', 'rb') as f:
   classifier = pickle.load(f)
   print('aaaa')
 
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
  if request.method == "POST":
    if request.files:
        image = request.files["image"]
        print(image)
        predict_it = app.config["IMAGE_UPLOADS"] + image.filename
        image.save(os.path.join(predict_it))
        print("Image saved")
        try:
          if predict_it != None:
              answer = predictImage(predict_it) 
              print('zzzzzzzzzz')
              return render_template("index.html", answer=answer)
        except:
          print("exept block")
          return redirect(request.url)
        # return render_template("index.html", answer='k')
  return render_template("index.html", answer='-')


@app.route('/success')
def welcome():
  image = request.args['image1']
  return render_template('saved.html')
 
def predictImage(path_to_image):
    img = cv2.imread(path_to_image,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    predict_this = np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    array = classifier.predict(predict_this)
    return labelling(array)

if __name__ == "__main__":
   app.run(threaded=False)