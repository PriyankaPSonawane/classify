from flask import Flask, request,  render_template, Response, redirect , url_for
import tensorflow as tf
from keras import backend as K
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

model = tf.keras.models.load_model("CNN.model")
model.load_weights("model.h5")
model._make_predict_function()

#from werkzeug import secure_filename
model1 = tf.keras.models.load_model("CNN_PNEMONIA.model")#model.save_weights("model.h5")
model1.load_weights("model_pnemonia.h5")
model1._make_predict_function()

#torch.save(model_conv,'cnn.pt')
the_model = torch.load('cnn.pt')
the_model.eval()

def generate_prediction(img):
      IMG_SIZE = 100
      img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
      img_array = img_array/255.0
      new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      input = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
      return input

def generate_prediction_p(img):
    sample_image = cv2.imread(img)
    sample_image = cv2.resize(sample_image, (224,224))
    if sample_image.shape[2] ==1:
        sample_image = np.dstack([sample_image, sample_image, sample_image])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = sample_image.astype(np.float32)/255.
    #sample_label = 1
    sample_image_processed = np.expand_dims(sample_image, axis=0)#since we pass only one image,we expand dim to include
                                                             #batch size 1
    return sample_image_processed

def generate_prediction_t(img):
    sample_image = Image.open(img)
    transform = transforms.Compose([            
                  transforms.Resize(256),                    
                  transforms.CenterCrop(224),                
                  transforms.ToTensor(),                     
                  transforms.Normalize(                      
                  mean=[0.485, 0.456, 0.406],                
                  std=[0.229, 0.224, 0.225]
                  )])
    img_t = transform(sample_image)
    return img_t
    

app = Flask(__name__)
#app.config["IMAGE_UPLOADS"] = os.getcwd()+"\\static"
@app.route('/')
def templates():
    return render_template('d.html')

@app.route('/Body_Segment')
def Body_Segment():
    return render_template('mypage.html')

@app.route('/Pneumonia_Detection')
def Pneumonia_Detection():
    return render_template('mypage1.html')

@app.route('/Tumor_Detection')
def Tumor_Detection():
    return render_template('mypage2.html')

@app.route('/results_for_segment', methods=['POST','GET'])
def results_for_segment():
      app.config["IMAGE_UPLOADS"] = os.getcwd()+"\\static"

      c = ["Brain", "Hands", "Kidney", "Legs", "Lungs", "Skull", "Teeth"]
      K.clear_session()
      if request.method == "POST":
          f = request.files['file']
          f.save(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          FOLDER = f.filename
          predi=generate_prediction(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          
          pred= model.predict(predi)
          predi = list(pred[0])
          #final = dict(zip(c, prediction)) 
          prediction = c[predi.index(max(predi))]
          accuracy = max(predi)
          accuracy = round((accuracy * 100),2)
          final_acc = str(accuracy) + "%"
          return render_template('mypage.html',image = FOLDER,prediction_text=prediction,prediction_acc = final_acc)

@app.route('/results_for_p', methods=['POST','GET'])
def results_for_p():
      app.config["IMAGE_UPLOADS"] = os.getcwd()+"\\static"
      K.clear_session()
      if request.method == "POST":
          f = request.files['file']
          f.save(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          FOLDER = f.filename
          predi=generate_prediction_p(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          prediction = model1.predict(predi)
          prediction = list(prediction[0])
          #final_acc= prediction.index(max(prediction))
          accuracy = max(prediction)
          accuracy = round((accuracy * 100),2)
          final_acc = str(accuracy) + "%"
          if accuracy < 65.00:
              prediction = "Normal"
          else:
              prediction = "Pneumonia"
      return render_template('mypage1.html',image1 = FOLDER,prediction_text=prediction,prediction_acc = final_acc)

@app.route('/results_for_t', methods=['POST','GET'])
def results_for_t():
      app.config["IMAGE_UPLOADS"] = os.getcwd()+"\\static"
      K.clear_session()
      if request.method == "POST":
          f = request.files['file']
          f.save(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          FOLDER = f.filename    
          #img_t = transform(app.config["IMAGE_UPLOADS"], f.filename)
          predi = generate_prediction_t(os.path.join(app.config["IMAGE_UPLOADS"], f.filename))
          batch_t = torch.unsqueeze(predi, 0)
          out = the_model(batch_t)
          #print(out.shape)
          class1 = ["No", "Yes" ]
          _, index = torch.max(out, 1)
          percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
          prediction = class1[index[0]]
          accuracy = percentage[index[0]].item()
          accuracy = round(accuracy,2)
          final_acc = str(accuracy) + "%"
          return render_template('mypage2.html',image2 = FOLDER,prediction_text=prediction,prediction_acc = final_acc)

if __name__ == '__main__':
    #app.debug = True
    app.run(host='192.168.8.25',port=5000)
    
