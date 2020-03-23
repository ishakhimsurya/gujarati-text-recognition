from flask import render_template, request, jsonify, Flask
import flask
import numpy as np
import traceback
import pickle
import cv2

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
          return("S")
        elif answer == 41:
          return("s")
        elif answer == 42:
          return("sh")
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
app = Flask(__name__)#template_folder='templates'
 
# importing models
with open('model/finalized_model.sav', 'rb') as f:
   classifier = pickle.load(f)
   print('aaaa')
 
@app.route('/')
def welcome():
   return "Boston Housing Price Prediction"
 
@app.route('/api/predict', methods=['POST','GET'])
def predict():
  
   if flask.request.method == 'GET':
       try:
        """img_loc = 'alien_test/Copy of 5ch-58.png'
        print("Img loc: ", img_loc)
        if img_loc != None: 
            img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            predict_this = np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
            # plt.imshow(img, cmap='gray')
            array = classifier.predict(predict_this)
            labelling(array)
            print('zzzzzzzzzz')
        return jsonify({
            "prediction1":str(labelling(array))
        })"""
        print("Hello")
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
 
   if flask.request.method == 'POST':
       try:
            # img_loc = 'alien_test/Copy of 5ch-58.png'
            # print("Img loc: ", img_loc)
            # if img_loc != None: 
            #     img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
            #     img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            #     predict_this = np.array(img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
            #     # plt.imshow(img, cmap='gray')
            #     array = classifier.predict(predict_this)
            #     labelling(array)
            # return jsonify({
            #     "prediction":str(labelling(array))
            # })
            print("hi")
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
      
 
if __name__ == "__main__":
   app.run(threaded=False)